### Lab 4, Part 3. Embeddings.

### This lab contains a number of steps in which you need to run existing
### code and evaluate how it works. You can paste it into a Python console or
### Run each block independently.
### You should include a PDF in your repo with answers to each of the questions.

## Requirements: numpy, matplotlib, scikit-learn, sentence-transformers

import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import os
import pandas as pd
from PIL import Image
from sklearn.datasets import fetch_openml
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sentence_transformers import SentenceTransformer
K = 3
SEED = 0


def cosine(a, b):
    """Cosine similarity between two vectors: 1 = same direction, 0 = unrelated."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

### Jaccard is similar to entropy: shared words / total distinct words
def word_overlap(s1, s2):
    w1 = set(s1.lower().replace(".", "").replace(",", "").split())
    w2 = set(s2.lower().replace(".", "").replace(",", "").split())
    return len(w1 & w2) / len(w1 | w2)


# %% Step 1 — Load a text embedding model from HuggingFace and try it out. This will
## take a bit to download the first time.


print("Loading text model (~90 MB the first time)...")
model = SentenceTransformer("all-MiniLM-L6-v2")

v = model.encode("The cat sat on the mat.")
print("Sentence in:   'The cat sat on the mat.'")
print("Embedding out:", v.shape)
print("First 10 numbers:", np.round(v[:10], 3))

### Question 1. In this part, we'll see how embeddings compare to Jaccard and cosine on three different
### types of sentences.


pairs = [
    # Same meaning, different words
    ("The dog chased the ball.", "A puppy ran after a toy."),
    ("I'm starving.", "I really need to eat something."),
    ("The film was fantastic.", "I loved that movie."),
    # Same words, different meaning
    ("I sat on the bank of the river.", "I went to the bank to get a loan."),
    ("The bat flew out of the cave.", "He swung the bat at the ball."),
    # Unrelated
    ("The dog chased the ball.", "Interest rates rose last quarter."),
    ("The cat sat on the mat.", "I really love ice cream")
]

print(f"\n{'Sentence 1':<36}{'Sentence 2':<38}{'Overlap':>8}{'Embed':>8}")
print("-" * 90)
for s1, s2 in pairs:
    e1, e2 = model.encode([s1, s2])
    print(f"{s1:<36}{s2:<38}{word_overlap(s1, s2):>8.2f}{cosine(e1, e2):>8.2f}")

### Before you run this, predict what you think is going to happen. Then compare your results
### to your prediction.

###  Which method scores the paraphrases higher?
###  What happens with "bank" and "bat"? Did the embedding model notice the different meanings?

### Question 2: Let's visualize how embeddings work with a heatmap. Notice that our examples here
### share almost no words.

sentences = [
    "The striker scored in the final minute.",
    "The team won the championship game.",
    "Preheat the oven to 350 degrees.",
    "Simmer the sauce for ten minutes.",
    "A storm is expected this weekend.",
    "It will be sunny and warm tomorrow.",
    "The new phone has a faster processor.",
    "Update your laptop's operating system.",
]
E = model.encode(sentences, normalize_embeddings=True)
S = E @ E.T  # with normalized vectors, the dot product IS cosine similarity

fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(S, cmap="viridis", vmin=0, vmax=1)
short = [s[:30] + ("..." if len(s) > 30 else "") for s in sentences]
ax.set_xticks(range(len(sentences)), short, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(sentences)), short, fontsize=8)
for i in range(len(sentences)):
    for j in range(len(sentences)):
        ax.text(j, i, f"{S[i, j]:.2f}", ha="center", va="center",
                color="white" if S[i, j] < 0.6 else "black", fontsize=8)
plt.colorbar(im, label="cosine similarity")
plt.title("Which sentences are close in embedding space?")
plt.tight_layout()
plt.show()

### Question 2:  You should see bright 2x2 blocks along the diagonal. What do they correspond to?


### Question 3: In this task, we'll compare the performance of bag-of-words and embeddings.

## Let's build a corpus with four categories of documents.

corpus = {
    "sports": [
        "The striker scored in the final minute.",
        "The team won the championship game.",
        "She broke the world record in the 100 meters.",
        "The coach called a timeout to discuss strategy.",
        "Fans cheered as the home side took the lead.",
        "He was injured during practice and will miss the season.",
        "The referee showed a red card after the foul.",
        "Our cyclist finished third in the mountain stage.",
        "The pitcher threw a no-hitter last night.",
        "Tickets for the playoff match sold out in minutes.",
        "The goalkeeper made an incredible save.",
        "They trained for months before the marathon.",
    ],
    "cooking": [
        "Preheat the oven to 350 degrees.",
        "Simmer the sauce for ten minutes.",
        "Chop the onions and garlic finely.",
        "Whisk the eggs with a pinch of salt.",
        "Let the dough rise in a warm place for an hour.",
        "Season the soup with fresh herbs before serving.",
        "Grill the vegetables until they are slightly charred.",
        "This recipe needs two cups of flour.",
        "Fry the tortillas until golden and crispy.",
        "Marinate the chicken overnight for the best flavor.",
        "Stir constantly so the risotto doesn't stick.",
        "The cake should cool before you add the frosting.",
    ],
    "weather": [
        "A storm is expected this weekend.",
        "It will be sunny and warm tomorrow.",
        "Heavy snow closed the mountain pass.",
        "The humidity makes it feel even hotter.",
        "Forecasters warned of flooding near the coast.",
        "Strong winds knocked down several trees.",
        "Morning fog should clear by noon.",
        "This has been the driest summer on record.",
        "Bring an umbrella, showers are likely this afternoon.",
        "Temperatures will drop below freezing tonight.",
        "A hurricane watch has been issued for the islands.",
        "The monsoon rains arrived two weeks late.",
    ],
    "technology": [
        "The new phone has a faster processor.",
        "Update your laptop's operating system.",
        "The company released a patch for the security bug.",
        "Cloud storage makes it easy to share files.",
        "The app crashed after the latest update.",
        "Engineers are building a more efficient battery.",
        "This router supports the newest wifi standard.",
        "The website went down due to heavy traffic.",
        "Restart the computer to finish installing.",
        "The startup trained a model to recognize speech.",
        "My password manager syncs across devices.",
        "The graphics card overheats during games.",
    ],
}
texts = [t for topic in corpus for t in corpus[topic]]
labels = np.array([topic for topic in corpus for _ in corpus[topic]])
TOPICS = list(corpus)
print(f"{len(texts)} sentences across {len(TOPICS)} topics")

tr, te = train_test_split(np.arange(len(texts)), test_size=0.5,
                          stratify=labels, random_state=SEED)

# Representation 1: "bag of words" — count how many times each word appears.
# We'll use the CountVectorizer from sklearn to do this. Note that we're not doing
# any filtering.

bow = CountVectorizer().fit([texts[i] for i in tr])
B_train = bow.transform([texts[i] for i in tr]).toarray()
B_test = bow.transform([texts[i] for i in te]).toarray()
print("Vocabulary size (bag-of-words features):", B_train.shape[1])

knn_bow = KNeighborsClassifier(n_neighbors=K, metric="cosine").fit(B_train, labels[tr])
acc_bow = accuracy_score(labels[te], knn_bow.predict(B_test))

## Representation 2: embeddings: Now let's embed the corpus.
## We use kNN, with K=3, for both representations.

E_corpus = model.encode(texts, normalize_embeddings=True)


knn_emb = KNeighborsClassifier(n_neighbors=K, metric="cosine").fit(E_corpus[tr], labels[tr])
pred_emb = knn_emb.predict(E_corpus[te])
acc_emb = accuracy_score(labels[te], pred_emb)

print(f"\nkNN on bag-of-words: {acc_bow:.0%}")
print(f"kNN on embeddings:   {acc_emb:.0%}")
print(f"Chance:              {1 / len(TOPICS):.0%}")

# Question 3: Copy the results into your PDF.

# Our corpora are not very large. Let's see how many words each test sentence shares with t
# the training vocabulary.
# Often it's only words like "the" and "a", which say nothing about the topic.

known = B_test.sum(axis=1)
print(f"\nKnown words per test sentence: min {known.min()}, median {int(np.median(known))}")
print("Example:", texts[te[0]], "-> known words:",
      [w for w, n in zip(bow.get_feature_names_out(), B_test[0]) if n > 0])

# Question 3: Why does bag-of-words struggle with a small corpus?
# Why is this not a problem for embeddings? (hint: How was the embedding function created?)
#



#### Question 4: Let's see where embeddings struggle.

tricky = [
    ("I love this movie.", "I do not love this movie."),
    ("The cat chased the dog.", "The dog chased the cat."),
    ("The meeting is at 3pm.", "The meeting is at 4pm."),
]
print(f"\n{'Sentence 1':<28}{'Sentence 2':<30}{'Embed':>8}")
print("-" * 66)
for s1, s2 in tricky:
    e1, e2 = model.encode([s1, s2])
    print(f"{s1:<28}{s2:<30}{cosine(e1, e2):>8.2f}")

# Question 4: Each pair means something different (sometimes the opposite!), yet scores high.
# What does "similar" really mean to this model? When would that be a problem?
# Would this be an issue for bag-of-words?


#### Part 2: Working with images.
#### In this part, you'll use the CLIP model to embed images and prompts involving the
### CIFAR-10 dataset. This is a collection of tiny (32x32) images, labeled into 10 categories.
### It's widely used for demonstrating image processing.
### Since we don't need to actually do any training, we're just going to get a subset of the data.
### It will still take a bit to download the first time through.


N_TRAIN = 5000
N_TEST = 1000
SEED = 0  # change to get a different random subset

### These are the 10 classes of images.
### Run the code below (- to the ##%%## mark) to load the CIFAR data

CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]
# Which OpenML copy of CIFAR-10 to fetch:
#   "CIFAR_10_small" — 20,000 images (about 1/3 the download; plenty for this lab)
#   "CIFAR_10"       — all 60,000 images
OPENML_NAME = "CIFAR_10_small"


def load_cifar_subset(n_train=N_TRAIN, n_test=N_TEST, seed=SEED, data_dir="data"):
    """Fetch CIFAR-10 via scikit-learn and return a balanced, reproducible subset.

    scikit-learn has no built-in CIFAR loader, but fetch_openml can download
    the "CIFAR_10" dataset from OpenML: 60,000 images, each stored as one row
    of 3,072 numbers (1,024 red values, then 1,024 green, then 1,024 blue).
    """
    cache = os.path.join(data_dir, f"cifar_openml_subset_{n_train}_{n_test}_{seed}.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        return d["X_train"], d["y_train"], d["X_test"], d["y_test"]

    os.makedirs(data_dir, exist_ok=True)
    print("Fetching CIFAR-10 from OpenML via scikit-learn (a few minutes, one time only)...")
    X_flat, y_str = fetch_openml(OPENML_NAME, version=1, as_frame=False,
                                 return_X_y=True, data_home=data_dir)

    # Each row is channel-first (R block, G block, B block). Reshape to
    # (N, 3, 32, 32), then move channels last to get (N, 32, 32, 3) for display.
    X_all = X_flat.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.uint8)
    y_all = np.asarray(y_str).astype(int)  # labels arrive as strings "0".."9"
    del X_flat

    # Pick two disjoint, class-balanced subsets: a collection and a test set.
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for c in range(10):
        idx = rng.permutation(np.where(y_all == c)[0])
        tr.append(idx[: n_train // 10])
        te.append(idx[n_train // 10 : n_train // 10 + n_test // 10])
    tr, te = rng.permutation(np.concatenate(tr)), rng.permutation(np.concatenate(te))

    out = dict(X_train=X_all[tr], y_train=y_all[tr], X_test=X_all[te], y_test=y_all[te])
    np.savez_compressed(cache, **out)
    return out["X_train"], out["y_train"], out["X_test"], out["y_test"]



# Our "unlabeled collection" is the 5,000 training images. We keep the labels
# in a clearly named variable and only use them for grading.
images, _HIDDEN_LABELS, X_test, y_test = load_cifar_subset()
print("Unlabeled collection:", images.shape)

# Quick look: these should be recognizable (if colors look scrambled, the
# reshape in the loader is wrong).
fig, axes = plt.subplots(2, 8, figsize=(12, 3.4))
for ax, img in zip(axes.flat, images[:16]):
    ax.imshow(img)
    ax.axis("off")
plt.suptitle("A few images from our unlabeled collection")
plt.tight_layout()
plt.show()

##%%## the above code will fetch and load the CIFAR-10 dataset.


# Question 5: Embedding text with CLIP. First embed the image collection with CLIP's IMAGE encoder
clip = SentenceTransformer("clip-ViT-B-32")

EMB_CACHE = f"data/clip_embeddings_openml_{N_TRAIN}_{N_TEST}_{SEED}.npz"
if os.path.exists(EMB_CACHE):
    d = np.load(EMB_CACHE)
    E_img, E_test = d["E_train"], d["E_test"]
    print("Loaded cached embeddings.")
else:
    print("Embedding images (will take a minute or so...")
    to_pil = lambda arr: [Image.fromarray(im) for im in arr]
    E_img = clip.encode(to_pil(images), batch_size=128, show_progress_bar=True)
    E_test = clip.encode(to_pil(X_test), batch_size=128, show_progress_bar=True)
    os.makedirs("data", exist_ok=True)
    np.savez_compressed(EMB_CACHE, E_train=E_img, E_test=E_test)

E_img = E_img / np.linalg.norm(E_img, axis=1, keepdims=True)
E_test = E_test / np.linalg.norm(E_test, axis=1, keepdims=True)
print("Image embeddings:", E_img.shape)

def embed_text(texts):
    t = clip.encode(texts if isinstance(texts, list) else [texts])
    return t / np.linalg.norm(t, axis=1, keepdims=True)


def text_search(query, k=8):
    """Return indices and scores of the k images most similar to the text."""
    sims = E_img @ embed_text(query)[0]
    top = np.argsort(-sims)[:k]
    return top, sims[top]


def show_search(queries, k=8):
    fig, axes = plt.subplots(len(queries), k, figsize=(1.8 * k, 2.1 * len(queries)))
    axes = np.atleast_2d(axes)
    for row, query in enumerate(queries):
        top, scores = text_search(query, k)
        for col, (j, s) in enumerate(zip(top, scores)):
            axes[row, col].imshow(images[j])
            axes[row, col].set_title(f"{s:.2f}", fontsize=8)
            axes[row, col].axis("off")
        axes[row, 0].text(-4, 16, f'"{query}"', ha="right", va="center", fontsize=10,
                          fontweight="bold")
    plt.suptitle("Text search over unlabeled images (number = similarity)")
    plt.tight_layout()
    plt.show()


show_search(["a red car", "a bird in the sky", "an animal in the snow",
             "a boat on the water", "a white horse"])

### Question 5: Add your own prompts as well. Don't use any of the words in the labels.
### How does it perform? How do you think this works?



# Question 6. — Grading the search with the hidden labels. Let's measure precision for the top
# k matches. As k increases, what happens to precision?

def precision_at_k(query, true_class, k=20):
    top, _ = text_search(query, k)
    return np.mean(_HIDDEN_LABELS[top] == true_class)


p10 = [precision_at_k(f"a photo of a {name}", c, k=10) for c, name in enumerate(CLASS_NAMES)]
p100 = [precision_at_k(f"a photo of a {name}", c, k=100) for c, name in enumerate(CLASS_NAMES)]

x = np.arange(len(CLASS_NAMES))
plt.figure(figsize=(10, 3.8))
plt.bar(x - 0.2, p10, width=0.4, label="precision@10")
plt.bar(x + 0.2, p100, width=0.4, label="precision@100")
plt.xticks(x, CLASS_NAMES, rotation=30)
plt.ylim(0, 1.05)
plt.ylabel("Fraction correct")
plt.title('Searching for "a photo of a {class}"')
plt.legend()
plt.tight_layout()
plt.show()

# Question 6: Precision usually drops from @10 to @100. Why do you think that is? Which classes
# are hardest to search for?


# Question 7. — Prompt engineering: Let's see if the phrasing in the prompt makes a difference.
templates = [
    "{}",
    "a photo of a {}",
    "a blurry photo of a {}",
    "a low resolution photo of a {}",
    "a drawing of a {}",
    "a photo of something",  # control: ignores the class name entirely!
]

template_acc = {}
for t in templates:
    E_t = embed_text([t.format(name) for name in CLASS_NAMES])
    template_acc[t] = accuracy_score(y_test, np.argmax(E_test @ E_t.T, axis=1))

# Ensemble: average the embeddings of several templates for each class
good = ["a photo of a {}", "a blurry photo of a {}", "a low resolution photo of a {}"]
E_ens = np.mean([embed_text([t.format(n) for n in CLASS_NAMES]) for t in good], axis=0)
E_ens = E_ens / np.linalg.norm(E_ens, axis=1, keepdims=True)
template_acc["ENSEMBLE of 3 templates"] = accuracy_score(
    y_test, np.argmax(E_test @ E_ens.T, axis=1))

print(f"\n{'Prompt template':<36}{'Accuracy':>10}")
print("-" * 46)
for t, a in template_acc.items():
    print(f"{t:<36}{a:>10.1%}")

## Question 7. Does modifying the prompt help?
## CIFAR images are tiny (32x32). Why might "a low resolution photo of a ..." help?
## Why does the control prompt score near 10%?

### Question 8: Wrap-up questions:
# a. What made it possible to compare a sentence with a picture?
# b. Who decided which words go near which images? (Think: where did CLIP's 400 million
#  captions come from?) What biases might that create?
# c. What kinds of queries failed? What kinds of IMAGES might CLIP handle
#    poorly because it rarely saw them?