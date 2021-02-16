import torch
from transformers import BertTokenizer, BertModel
import config

device = config.device

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
model.eval()

def join(tokens):
    res = []
    for token in tokens:
        if token[:2] == '##':
            res.append(token[2:])
        else:
            res.append(token)
    return ''.join(res)


model.to(device)
def embed(text):
    marked_text = "[CLS] " + ' '.join(text) + " [SEP]"

    tokens = tokenizer.tokenize(marked_text)
    seg = [1] * len(tokens)
    indices = tokenizer.convert_tokens_to_ids(tokens)

    tokens_t = torch.tensor([indices]).to(device)
    seg_t = torch.tensor([seg]).to(device)
    with torch.no_grad():
        outputs = model(tokens_t, seg_t)
        embeddings = outputs[2][-2][0]  # second to last layer
    res = []
    n = 0
    i = 1
    while n < len(text):
        j = i + 1
        while join(tokens[i:j]) != text[n].lower():
            j += 1
        if j == i + 1:
            res.append(embeddings[i])
        else:
            res.append(torch.mean(embeddings[i:j], dim=0))
        i = j
        n += 1
    res = [embedding.cpu().detach().numpy() for embedding in res]
    return res
