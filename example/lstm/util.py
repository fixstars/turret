import torch

id_to_char = {}
char_to_id = {}


def _update_vocab(txt):
    chars = list(txt)

    for i, char in enumerate(chars):
        if char not in char_to_id:
            tmp_id = len(char_to_id)
            char_to_id[char] = tmp_id
            id_to_char[tmp_id] = char


def load_text(file_name, use_dict=False, dict_data=None):
    with open(file_name, 'r') as f:
        txt_list = f.readlines()

    questions, answers = [], []

    for txt in txt_list:
        idx = txt.find('_')
        questions.append(txt[:idx])
        answers.append(txt[idx:-1])

    # create vocab dict
    if use_dict is False:
        for i in range(len(questions)):
            q, a = questions[i], answers[i]
            _update_vocab(q)
            _update_vocab(a)

    # create torch array
    x = torch.zeros([len(questions), len(questions[0]), 1], dtype=torch.long,
                    device=get_device())
    t = torch.zeros([len(questions), len(answers[0]), 1], dtype=torch.long,
                    device=get_device())

    if use_dict is False:
        vocab_dict = char_to_id
    else:
        vocab_dict = dict_data

    for i, sentence in enumerate(questions):
        x[i, :, 0] = torch.Tensor([vocab_dict[c] for c in list(sentence)])
    for i, sentence in enumerate(answers):
        t[i, :, 0] = torch.Tensor([vocab_dict[c] for c in list(sentence)])

    return (x, t)


def get_id_from_char(c):
    return char_to_id[c]


def get_max_dict():
    return len(char_to_id)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dict_id_to_char():
    return id_to_char


def get_dict_char_to_id():
    return char_to_id
