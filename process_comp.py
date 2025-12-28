import argparse
import collections
import gzip
import html
import json
import os
import random
import re
import torch
from tqdm import tqdm

from UniSRec.dataset.preprocessing.utils import check_path, set_device, load_plm, amazon_dataset2fullname


def load_ratings(file):
    users, items, inters = set(), set(), set()
    with open(file, 'r') as fp:
        for line in tqdm(fp, desc='Load ratings'):
            try:
                item, user, rating, time = line.strip().split(',')
                users.add(user)
                items.add(item)
                inters.add((user, item, float(rating), int(time)))
            except ValueError:
                print(line)
    return users, items, inters


def load_meta_items(file):
    items = set()
    with gzip.open(file, 'r') as fp:
        for line in tqdm(fp, desc='Load metas'):
            data = json.loads(line)
            items.add(data['asin'])
    return items


def get_user2count(inters):
    user2count = collections.defaultdict(int)
    for unit in inters:
        user2count[unit[0]] += 1
    return user2count


def get_item2count(inters):
    item2count = collections.defaultdict(int)
    for unit in inters:
        item2count[unit[1]] += 1
    return item2count


def generate_candidates(unit2count, threshold):
    cans = set()
    for unit, count in unit2count.items():
        if count >= threshold:
            cans.add(unit)
    return cans, len(unit2count) - len(cans)


def filter_inters(inters, can_items=None,
                  user_k_core_threshold=0, item_k_core_threshold=0):
    new_inters = []

    # filter by meta items
    if can_items:
        print('\nFiltering by meta items: ')
        for unit in inters:
            if unit[1] in can_items:
                new_inters.append(unit)
        inters, new_inters = new_inters, []
        print('    The number of inters: ', len(inters))

    # filter by k-core
    if user_k_core_threshold or item_k_core_threshold:
        print('\nFiltering by k-core:')
        idx = 0
        user2count = get_user2count(inters)
        item2count = get_item2count(inters)

        while True:
            new_user2count = collections.defaultdict(int)
            new_item2count = collections.defaultdict(int)
            users, n_filtered_users = generate_candidates(
                user2count, user_k_core_threshold)
            items, n_filtered_items = generate_candidates(
                item2count, item_k_core_threshold)
            if n_filtered_users == 0 and n_filtered_items == 0:
                break
            for unit in inters:
                if unit[0] in users and unit[1] in items:
                    new_inters.append(unit)
                    new_user2count[unit[0]] += 1
                    new_item2count[unit[1]] += 1
            idx += 1
            inters, new_inters = new_inters, []
            user2count, item2count = new_user2count, new_item2count
            print('    Epoch %d The number of inters: %d, users: %d, items: %d'
                    % (idx, len(inters), len(user2count), len(item2count)))
    return inters


def make_inters_in_order(inters):
    user2inters, new_inters = collections.defaultdict(list), list()
    for inter in inters:
        user, item, rating, timestamp = inter
        user2inters[user].append((user, item, rating, timestamp))
    for user in user2inters:
        user_inters = user2inters[user]
        user_inters.sort(key=lambda d: d[3])
        for inter in user_inters:
            new_inters.append(inter)
    return new_inters


def preprocess_rating(args):
    dataset_full_name = amazon_dataset2fullname[args.dataset]

    print('Process rating data: ')
    print(' Dataset: ', dataset_full_name)

    # load ratings
    rating_file_path = os.path.join(args.input_path, 'Ratings', dataset_full_name + '.csv')
    rating_users, rating_items, rating_inters = load_ratings(rating_file_path)

    # load item IDs with meta data
    meta_file_path = os.path.join(args.input_path, 'Metadata', f'meta_{dataset_full_name}.json.gz')
    meta_items = load_meta_items(meta_file_path)

    # 1. Filter items w/o meta data;
    # 2. K-core filtering;
    print('The number of raw inters: ', len(rating_inters))
    rating_inters = filter_inters(rating_inters, can_items=meta_items,
                                  user_k_core_threshold=args.user_k,
                                  item_k_core_threshold=args.item_k)

    # sort interactions chronologically for each user
    rating_inters = make_inters_in_order(rating_inters)
    print('\n')

    # return: list of (user_ID, item_ID, rating, timestamp)
    return rating_inters


def get_user_item_from_ratings(ratings):
    users, items = set(), set()
    for line in ratings:
        user, item, rating, time, history = line
        users.add(user)
        items.add(item)
    return users, items


def clean_text(raw_text):
    if isinstance(raw_text, list):
        cleaned_text = ' '.join(raw_text)
    elif isinstance(raw_text, dict):
        cleaned_text = str(raw_text)
    else:
        cleaned_text = raw_text
    cleaned_text = html.unescape(cleaned_text)
    cleaned_text = re.sub(r'["\n\r]*', '', cleaned_text)
    index = -1
    while -index < len(cleaned_text) and cleaned_text[index] == '.':
        index -= 1
    index += 1
    if index == 0:
        cleaned_text = cleaned_text + '.'
    else:
        cleaned_text = cleaned_text[:index] + '.'
    if len(cleaned_text) >= 2000:
        cleaned_text = ''
    return cleaned_text


def generate_text(args, items):
    # load metadata file (CSV mode already handled earlier)
    if args.input_format == 'csv':
        meta_file_path = os.path.join(
            args.input_path,
            'meta_All_Beauty.jsonl'
        )
    else:
        dataset_full_name = amazon_dataset2fullname[args.dataset]
        meta_file_path = os.path.join(
            args.input_path,
            'Metadata',
            f'meta_{dataset_full_name}.json.gz'
        )

    print("Load metadata from:", meta_file_path)

    # read metadata
    item_meta = {}
    open_fn = gzip.open if meta_file_path.endswith('.gz') else open
    with open_fn(meta_file_path, 'rt') as f:
        for line in f:
            obj = json.loads(line)
            if 'parent_asin' in obj:
                item_meta[obj['parent_asin']] = obj

    def flatten(x):
        if x is None:
            return []
        if isinstance(x, str):
            return [x]
        if isinstance(x, list):
            out = []
            for v in x:
                out.extend(flatten(v))
            return out
        return []

    item_text = {}
    for item in items:
        meta = item_meta.get(item, {})
        parts = []

        # THESE KEYS EXIST IN *YOUR* FILE — NO GUESSING
        parts += flatten(meta.get('title'))
        parts += flatten(meta.get('main_category'))
        parts += flatten(meta.get('categories'))
        parts += flatten(meta.get('store'))
        parts += flatten(meta.get('features'))
        parts += flatten(meta.get('description'))

        text = ' '.join(parts)
        item_text[item] = text

    return list(item_text.items())

def load_text(file):
    item_text_list = []
    with open(file, 'r') as fp:
        fp.readline()
        for line in fp:
            try:
                item, text = line.strip().split('\t', 1)
            except ValueError:
                item = line.strip()
                text = '.'
            item_text_list.append([item, text])
    return item_text_list


def write_text_file(item_text_list, file):
    print('Writing text file: ')
    with open(file, 'w') as fp:
        fp.write('item_id:token\ttext:token_seq\n')
        for item, text in item_text_list:
            fp.write(item + '\t' + text + '\n')


def preprocess_text(args, rating_inters):
    print('Process text data: ')
    print(' Dataset: ', args.dataset)
    rating_users, rating_items = get_user_item_from_ratings(rating_inters)

    # load item text and clean
    item_text_list = generate_text(args, rating_items)
    print('\n')

    # return: list of (item_ID, cleaned_item_text)
    return item_text_list


def convert_inters2dict(inters):
    user2items = collections.defaultdict(list)
    user2index, item2index = dict(), dict()

    for user, item, rating, timestamp in inters:
        if user not in user2index:
            user2index[user] = len(user2index)
        if item not in item2index:
            item2index[item] = len(item2index)

        user2items[user2index[user]].append(
            (item2index[item], timestamp)
        )

    return user2items, user2index, item2index

def filter_users_by_min_interactions(rating_inters, min_k=2):
    from collections import defaultdict

    user2inters = defaultdict(list)
    for inter in rating_inters:
        user = inter[0]
        user2inters[user].append(inter)

    kept_users = {u for u, v in user2inters.items() if len(v) >= min_k}

    filtered = [inter for inter in rating_inters if inter[0] in kept_users]

    print(f"Filtered users: kept {len(kept_users)} users with >= {min_k} interactions")
    return filtered


# def generate_training_data(args, rating_inters, train_inter_num, valid_inter_num):
#     print('Split dataset: ')
#     print(' Dataset: ', args.dataset)
#     print(" Rating_inters:", len(rating_inters), type(rating_inters))

#     # generate train valid test
#     user2items, user2index, item2index = convert_inters2dict(rating_inters)
#     train_inters, valid_inters, test_inters = dict(), dict(), dict()
#     for u_index in range(len(user2index)):
#         inters = user2items[u_index]

#         train_inters[u_index] = inters[:-2]
#         valid_inters[u_index] = inters[-2]
#         test_inters[u_index] = inters[-1]
#     return train_inters, valid_inters, test_inters, user2index, item2index

import collections

# def generate_training_data(args, rating_inters, train_inter_num, valid_inter_num):
#     print('Split dataset:')
#     print(' Dataset:', args.dataset)

#     # ---------- split by original dataset boundary ----------
#     train_raw = rating_inters[:train_inter_num]
#     valid_raw = rating_inters[train_inter_num:train_inter_num + valid_inter_num]
#     test_raw  = rating_inters[train_inter_num + valid_inter_num:]

#     # ---------- index mapping ----------
#     user2index = {}
#     item2index = {}

#     def get_uid(u):
#         if u not in user2index:
#             user2index[u] = len(user2index)
#         return user2index[u]

#     def get_iid(i):
#         if i not in item2index:
#             item2index[i] = len(item2index)
#         return item2index[i]

#     # ---------- containers ----------
#     train_inters = collections.defaultdict(list)
#     valid_inters = collections.defaultdict(list)
#     test_inters  = collections.defaultdict(list)

#     # ---------- helper ----------
#     def add_to(split_dict, inters):
#         for user, item, rating, timestamp in inters:
#             u_idx = get_uid(user)
#             i_idx = get_iid(item)
#             ts = float(timestamp)
#             # split_dict[u_idx].append((i_idx, ts))
#             split_dict[user].append((item, ts))

#     # ---------- populate ----------
#     add_to(train_inters, train_raw)
#     add_to(valid_inters, valid_raw)
#     add_to(test_inters,  test_raw)

#     # ---------- sort by timestamp per user (VERY IMPORTANT) ----------
#     for d in (train_inters, valid_inters, test_inters):
#         for u in d:
#             d[u].sort(key=lambda x: x[1])

#     return train_inters, valid_inters, test_inters, user2index, item2index

def generate_training_data(args, rating_inters, train_inter_num, valid_inter_num):
    print('Split dataset:')
    print(' Dataset:', args.dataset)

    # ---------- split ----------
    train_raw = rating_inters[:train_inter_num]
    valid_raw = rating_inters[train_inter_num:train_inter_num + valid_inter_num]
    test_raw  = rating_inters[train_inter_num + valid_inter_num:]

    # ---------- index mapping ----------
    user2index = {}
    item2index = {}

    def get_uid(u):
        if u not in user2index:
            user2index[u] = len(user2index)
        return user2index[u]

    def get_iid(i):
        if i not in item2index:
            item2index[i] = len(item2index)
        return item2index[i]

    # ---------- containers ----------
    # train / valid: user -> list of (history_items, target_item)
    train_inters = collections.defaultdict(list)
    valid_inters = collections.defaultdict(list)

    # test: user -> list of history_items
    test_inters  = collections.defaultdict(list)

    # ---------- helpers ----------
    def parse_history(h):
        if isinstance(h, str):
            return h.split()
        return list(h)

    # ---------- populate train ----------
    for row in train_raw:
        # expected: user, parent_asin, rating, timestamp, history
        user, target, _, _, history = row

        get_uid(user)
        hist_items = parse_history(history)

        # index everything
        for i in hist_items:
            get_iid(i)
        get_iid(target)

        train_inters[user].append((hist_items, target))

    # ---------- populate valid ----------
    for row in valid_raw:
        user, target, _, _, history = row

        get_uid(user)
        hist_items = parse_history(history)

        for i in hist_items:
            get_iid(i)
        get_iid(target)

        valid_inters[user].append((hist_items, target))

    # ---------- populate test ----------
    for row in test_raw:
        # expected: user, history   (adjust if schema differs)
        user, last_item, rating, ts, history = row

        get_uid(user)
        hist_items = parse_history(history)

        for i in hist_items:
            get_iid(i)

        test_inters[user].append(hist_items)

    return train_inters, valid_inters, test_inters, user2index, item2index


def load_unit2index(file):
    unit2index = dict()
    with open(file, 'r') as fp:
        for line in fp:
            unit, index = line.strip().split('\t')
            unit2index[unit] = int(index)
    return unit2index


def write_remap_index(unit2index, file):
    with open(file, 'w') as fp:
        for unit in unit2index:
            fp.write(unit + '\t' + str(unit2index[unit]) + '\n')


def generate_item_embedding(args, item_text_list, item2index, tokenizer, model, word_drop_ratio=-1):
    print(f'Generate Text Embedding by {args.emb_type}: ')
    print(' Dataset: ', args.dataset)

    items, texts = zip(*item_text_list)
    order_texts = [[0]] * len(items)
    for item, text in item_text_list:
        if item not in item2index:
            continue
        order_texts[item2index[item]] = text
    for item, idx in item2index.items():
        if order_texts[idx] is None:
            order_texts[idx] = ""
    for text in order_texts:
        # assert text != [0]
        if text == [0]:
            text = ""

    embeddings = []
    start, batch_size = 0, 64
    while start < len(order_texts):
        # sentences = order_texts[start: start + batch_size]
        sentences = ["" if s is None else str(s) for s in order_texts[start: start + batch_size]]
        if word_drop_ratio > 0:
            print(f'Word drop with p={word_drop_ratio}')
            new_sentences = []
            for sent in sentences:
                new_sent = []
                sent = sent.split(' ')
                for wd in sent:
                    rd = random.random()
                    if rd > word_drop_ratio:
                        new_sent.append(wd)
                new_sent = ' '.join(new_sent)
                new_sentences.append(new_sent)
            sentences = new_sentences
        bad = [(i, type(s), s) for i, s in enumerate(sentences) if not isinstance(s, str)]
        if bad:
            print("NON-STRING SENTENCES (should be impossible now):", bad[:3])
            raise RuntimeError("Sanitization failed")
        encoded_sentences = tokenizer(sentences, padding=True, max_length=512,
                                      truncation=True, return_tensors='pt').to(args.device)
        outputs = model(**encoded_sentences)
        if args.emb_type == 'CLS':
            cls_output = outputs.last_hidden_state[:, 0, ].detach().cpu()
            embeddings.append(cls_output)
        elif args.emb_type == 'Mean':
            masked_output = outputs.last_hidden_state * encoded_sentences['attention_mask'].unsqueeze(-1)
            mean_output = masked_output[:,1:,:].sum(dim=1) / \
                encoded_sentences['attention_mask'][:,1:].sum(dim=-1, keepdim=True)
            mean_output = mean_output.detach().cpu()
            embeddings.append(mean_output)
        start += batch_size
    embeddings = torch.cat(embeddings, dim=0).numpy()
    print('Embeddings shape: ', embeddings.shape)

    # suffix=1, output DATASET.feat1CLS, with word drop ratio 0;
    # suffix=2, output DATASET.feat2CLS, with word drop ratio > 0;
    if word_drop_ratio > 0:
        suffix = '2'
    else:
        suffix = '1'

    file = os.path.join(args.output_path, args.dataset,
                        args.dataset + '.feat' + suffix + args.emb_type)
    embeddings.tofile(file)


# def convert_to_atomic_files(args, train_data, valid_data, test_data):
#     print('Convert dataset: ')
#     print(' Dataset: ', args.dataset)
#     def write_inter_file(path, inter_dict):
#         with open(path, "w", encoding="utf-8") as f:
#             f.write("user_id:token\titem_id:token\ttimestamp:float\n")
#             for u_idx, lst in inter_dict.items():
#                 for i_idx, ts in lst:
#                     f.write(f"{u_idx}\t{i_idx}\t{ts}\n")
#     write_inter_file(os.path.join(args.output_path, args.dataset, f'{args.dataset}.train.inter'), train_data)
#     write_inter_file(os.path.join(args.output_path, args.dataset, f'{args.dataset}.valid.inter'), valid_data)
#     write_inter_file(os.path.join(args.output_path, args.dataset, f'{args.dataset}.test.inter'), test_data)

def convert_to_atomic_files(
    args,
    train_inters,   # dict[user] -> list[(history_items, target_item)]
    valid_inters,   # dict[user] -> list[(history_items, target_item)]
    test_inters,    # dict[user] -> list[history_items]
    max_len=50
):
    import os

    print("Convert dataset:")
    print(" Dataset:", args.dataset)

    out_dir = os.path.join(args.output_path, args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    HEADER = "user_id:token\titem_id_list:token_seq\titem_id:token\n"

    def trim(hist):
        return hist[-max_len:]

    # ======================================================
    # TRAIN
    # ======================================================
    with open(os.path.join(out_dir, f"{args.dataset}.train.inter"), "w", encoding="utf-8") as f:
        f.write(HEADER)

        for user, interactions in train_inters.items():
            for history, target in interactions:
                history = trim(history)
                if not history:
                    continue

                f.write(
                    f"{user}\t"
                    f"{' '.join(history)}\t"
                    f"{target}\n"
                )

    # ======================================================
    # VALID
    # ======================================================
    with open(os.path.join(out_dir, f"{args.dataset}.valid.inter"), "w", encoding="utf-8") as f:
        f.write(HEADER)

        for user, interactions in valid_inters.items():
            for history, target in interactions:
                history = trim(history)
                if not history:
                    continue

                f.write(
                    f"{user}\t"
                    f"{' '.join(history)}\t"
                    f"{target}\n"
                )

    # ======================================================
    # BUILD (vocab / embedding coverage only)
    # ======================================================
    with open(os.path.join(out_dir, f"{args.dataset}.build.inter"), "w", encoding="utf-8") as f:
        f.write(HEADER)

        # ---- include train + valid (supervised) ----
        for split in (train_inters, valid_inters):
            for user, interactions in split.items():
                for history, target in interactions:
                    history = trim(history)
                    if not history:
                        continue

                    f.write(
                        f"{user}\t"
                        f"{' '.join(history)}\t"
                        f"{target}\n"
                    )

        # ---- include kaggle test (history only) ----
        for user, histories in test_inters.items():
            for history in histories:
                history = trim(history)
                if not history:
                    continue

                pseudo_target = history[-1]           # safe: already observed
                hist2 = history[:-1] if len(history) > 1 else history

                f.write(
                    f"{user}\t"
                    f"{' '.join(hist2)}\t"
                    f"{pseudo_target}\n"
                )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Scientific', help='Pantry / Scientific / Instruments / Arts / Office')
    parser.add_argument('--user_k', type=int, default=5, help='user k-core filtering')
    parser.add_argument('--item_k', type=int, default=5, help='item k-core filtering')
    parser.add_argument('--input_path', type=str, default='../raw/')
    parser.add_argument('--output_path', type=str, default='../downstream/')
    parser.add_argument('--gpu_id', type=int, default=0, help='ID of running GPU')
    parser.add_argument('--plm_name', type=str, default='bert-base-uncased')
    parser.add_argument('--emb_type', type=str, default='CLS', help='item text emb type, can be CLS or Mean')
    parser.add_argument('--word_drop_ratio', type=float, default=-1, help='word drop ratio, do not drop by default')
    parser.add_argument('--input_format', type=str, default='csv',
                    choices=['amazon', 'csv'],
                    help='Input data format')
    return parser.parse_args()

def load_interactions_from_csv(file_path):
    inters = []
    with open(file_path, 'r') as f:
        header = f.readline().strip().split(',')
        col = {k: i for i, k in enumerate(header)}

        for line in tqdm(f, desc=f'Load {os.path.basename(file_path)}'):
            parts = line.strip().split(',')
            user = parts[col['user_id']]
            item = parts[col['parent_asin']]
            time = int(parts[col['timestamp']])
            seq = parts[col['history']]
            inters.append((user, item, 1.0, time, seq))
    return inters

def load_interactions_from_test_csv(file_path):
    inters = []
    with open(file_path, 'r') as f:
        header = f.readline().strip().split(',')
        col = {k: i for i, k in enumerate(header)}

        for line in tqdm(f, desc=f'Load {os.path.basename(file_path)}'):
            parts = line.strip().split(',')
            history = parts[col['history']]
            items = history.strip().split(' ')
            user = parts[col['id']]
            inters.append((user, items[-1], 1.0, len(items), history))
            # for ind, item in enumerate(items):
            #     time = ind + 1
            #     inters.append((user, item, 1.0, time))
    return inters

def _flatten_text(x):
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    if isinstance(x, list):
        out = []
        for v in x:
            out.extend(_flatten_text(v))
        return out
    return []


if __name__ == '__main__':
    args = parse_args()

    # load interactions from raw rating file
    if args.input_format == 'csv':
        train_inters = load_interactions_from_csv(
            os.path.join(args.input_path, f'{args.dataset}.train.csv')
        )
        print("train interactions sample:", train_inters[:2])
        valid_inters = load_interactions_from_csv(
            os.path.join(args.input_path, f'{args.dataset}.valid.csv')
        )
        test_inters = load_interactions_from_test_csv(
            os.path.join(args.input_path, f'{args.dataset}.test.csv')
        )
        rating_inters = train_inters + valid_inters + test_inters
        train_inter_num = len(train_inters)
        valid_inter_num = len(valid_inters)

    else:
        rating_inters = preprocess_rating(args)
    # load item text from raw meta data file
    item_text_list = preprocess_text(args, rating_inters)
    print("item_text_list sample:", item_text_list[:2])
    # rating_inters = filter_users_by_min_interactions(rating_inters, min_k=2)

    # split train/valid/test
    train_inters, valid_inters, test_inters, user2index, item2index = \
        generate_training_data(args, rating_inters, train_inter_num, valid_inter_num)
    print("train_inters sample:", list(train_inters.items())[:2])
    print("Train users:", len(train_inters))
    print("Avg interactions per train user:",
        sum(len(v) for v in train_inters.values()) / len(train_inters))
    # device & plm initialization
    device = set_device(args.gpu_id)
    args.device = device
    plm_tokenizer, plm_model = load_plm(args.plm_name)
    plm_model = plm_model.to(device)

    # create output dir
    check_path(os.path.join(args.output_path, args.dataset))


    # save interaction sequences into atomic files
    convert_to_atomic_files(args, train_inters, valid_inters, test_inters)

    # save useful data
    write_text_file(item_text_list, os.path.join(args.output_path, args.dataset, f'{args.dataset}.text'))
    print("user2index sample:", list(user2index.items())[:2])
    print("item2index sample:", list(item2index.items())[:2])
    write_remap_index(user2index, os.path.join(args.output_path, args.dataset, f'{args.dataset}.user2index'))
    write_remap_index(item2index, os.path.join(args.output_path, args.dataset, f'{args.dataset}.item2index'))

    # generate PLM emb and save to file
    generate_item_embedding(args, item_text_list, item2index, 
                            plm_tokenizer, plm_model, word_drop_ratio=-1)
    # pre-stored word drop PLM embs
    if args.word_drop_ratio > 0:
        generate_item_embedding(args, item_text_list, item2index, 
                                plm_tokenizer, plm_model, word_drop_ratio=args.word_drop_ratio)
