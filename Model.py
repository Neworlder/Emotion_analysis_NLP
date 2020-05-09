from gensim.models import word2vec
with open('sina/sinanews.train', 'r', encoding='utf-8') as f:
    lines = f.readlines()
with open('sina/sinanews.test', 'r', encoding='utf-8') as f:
    lines += f.readlines()
for i, news in enumerate(lines):
    lines[i] = news.split()[10:]
model = word2vec.Word2Vec(lines)
model.save('Model')
