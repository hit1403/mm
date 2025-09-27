"""
Latent Semantic Indexing (LSI) with SVD
"""

import numpy as np

# ------------------------------
# Step 1: Preprocess docs → term-document matrix
# ------------------------------
def build_term_doc_matrix(docs):
    tokenized = [doc.lower().split() for doc in docs]
    vocab = sorted(set(word for doc in tokenized for word in doc))

    A = np.zeros((len(vocab), len(docs)))
    for j, doc in enumerate(tokenized):
        for word in doc:
            i = vocab.index(word)
            A[i, j] += 1
    return A, vocab


# ------------------------------
# Step 2: Apply SVD and reduce dimensions
# ------------------------------
def lsi(A, k=2):
    U, S, VT = np.linalg.svd(A, full_matrices=False)
    Uk = U[:, :k]
    Sk = np.diag(S[:k])
    Vk = VT[:k, :]
    return U, S, VT, Uk, Sk, Vk


# ------------------------------
# Step 3: Project query into reduced space
# ------------------------------
def project_query(query, vocab, Uk, Sk):
    q_vec = np.zeros((len(vocab), 1))
    for word in query.lower().split():
        if word in vocab:
            i = vocab.index(word)
            q_vec[i, 0] += 1
    q_reduced = np.dot(np.linalg.inv(Sk), np.dot(Uk.T, q_vec))
    return q_vec, q_reduced


# ------------------------------
# Step 4: Search
# ------------------------------
def search_lsi(query, docs, k=2, top_k=3):
    A, vocab = build_term_doc_matrix(docs)
    print("=== Term-Document Matrix A ===")
    print(A)
    print("Vocab:", vocab, "\n")

    U, S, VT, Uk, Sk, Vk = lsi(A, k)
    print("=== Full U Matrix ===")
    print(U)
    print("\n=== Singular Values Σ ===")
    print(S)
    print("\n=== Full V^T Matrix ===")
    print(VT)

    print("\n=== Reduced Uk ===")
    print(Uk)
    print("\n=== Reduced Sk ===")
    print(Sk)
    print("\n=== Reduced Vk ===")
    print(Vk, "\n")

    # Project query
    q_vec, q_reduced = project_query(query, vocab, Uk, Sk)
    print("=== Original Query Vector ===")
    print(q_vec.flatten())
    print("\n=== Reduced Query Vector ===")
    print(q_reduced.flatten(), "\n")

    # Document vectors in reduced space
    doc_reduced = np.dot(Sk, Vk)
    print("=== Document Vectors in Reduced Space ===")
    print(doc_reduced, "\n")

    # Cosine similarity
    sims = []
    for i in range(doc_reduced.shape[1]):
        d = doc_reduced[:, i]
        sim = np.dot(q_reduced.flatten(), d) / (
            np.linalg.norm(q_reduced) * np.linalg.norm(d)
        )
        sims.append((i, sim))
        print(f"Cosine similarity(Query, Doc{i}) = {sim:.3f}")

    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:top_k]



docs = [
    "information retrieval is the process of obtaining information",
    "machine learning is about learning from data",
    "search engines use information retrieval models",
    "data science uses machine learning techniques"
]

query = "information system"
results = search_lsi(query, docs, k=2, top_k=3)

print("\n=== Final Results ===")
print("Query:", query)
for doc_id, score in results:
    print(f"Doc {doc_id}: {docs[doc_id]} (score={score:.3f})")



# ==========================================================================================

import random

# ------------------------------
# Step 1: Shingling
# ------------------------------
def get_shingles(doc, k=3):
    words = doc.lower().split()
    shingles = set()
    for i in range(len(words) - k + 1):
        shingle = " ".join(words[i:i+k])
        shingles.add(shingle)
    return shingles

# ------------------------------
# Step 2: Jaccard Similarity
# ------------------------------
def jaccard(set1, set2):
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)

# ------------------------------
# Step 3: MinHash (Random Permutations)
# ------------------------------
def minhash_permutation(shingle_sets, num_perm=10):
    all_shingles = list(set().union(*shingle_sets))
    N = len(all_shingles)

    # assign IDs
    shingle_to_id = {s: i for i, s in enumerate(all_shingles)}

    signatures = []
    for doc_set in shingle_sets:
        sig = []
        doc_ids = {shingle_to_id[s] for s in doc_set}
        for _ in range(num_perm):
            perm = list(range(N))
            random.shuffle(perm)
            # first shingle in permutation that belongs to doc
            for idx in perm:
                if idx in doc_ids:
                    sig.append(idx)
                    break
        signatures.append(sig)
    return signatures, all_shingles

# ------------------------------
# Step 4: MinHash (Hash Functions)
# ------------------------------
def hash_functions(num_hashes=None, max_val=None, given_funcs=None):
    funcs = []
    if given_funcs:
        funcs = given_funcs
    else:
        for _ in range(num_hashes):
            a = random.randint(1, max_val - 1)
            b = random.randint(0, max_val - 1)
            funcs.append(lambda x, a=a, b=b: (a*x + b) % max_val)
    return funcs

def minhash_hashing(shingle_sets, num_hashes=10, given_funcs=None):
    all_shingles = list(set().union(*shingle_sets))
    N = len(all_shingles)
    shingle_to_id = {s: i for i, s in enumerate(all_shingles)}

    # Use given hash functions if provided, else random
    if given_funcs:
        funcs = given_funcs
    else:
        funcs = hash_functions(num_hashes=num_hashes, max_val=N*2)

    signatures = []
    for doc_set in shingle_sets:
        sig = []
        ids = [shingle_to_id[s] for s in doc_set]
        for h in funcs:
            sig.append(min(h(i) for i in ids))
        signatures.append(sig)
    return signatures, all_shingles

# ------------------------------
# Step 5: Compare signatures
# ------------------------------
def signature_similarity(sig1, sig2):
    matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
    return matches / len(sig1)


# ------------------------------
# MAIN DEMO
# ------------------------------
if __name__ == "__main__":
    docs = [
        "information retrieval is the process of obtaining information",
        "search engines use information retrieval models",
        "machine learning is about learning from data",
        "information retrieval system for search"
    ]

    k = 2  # size of shingles
    shingle_sets = [get_shingles(doc, k) for doc in docs]

    print("=== Shingles (k=%d) ===" % k)
    for i, s in enumerate(shingle_sets):
        print(f"D{i}: {s}")
    print()

    # --- Exact Jaccard ---
    print("=== Exact Jaccard Similarities ===")
    for i in range(len(docs)):
        for j in range(i+1, len(docs)):
            sim = jaccard(shingle_sets[i], shingle_sets[j])
            print(f"D{i} vs D{j}: {sim:.3f}")
    print()

    # --- MinHash with Random Permutations ---
    perm_sigs, all_shingles_perm = minhash_permutation(shingle_sets, num_perm=10)
    print("=== MinHash with Random Permutations ===")
    print("All unique shingles:", all_shingles_perm, "\n")
    for i, sig in enumerate(perm_sigs):
        print(f"D{i} signature: {sig}")
    print()
    for i in range(len(docs)):
        for j in range(i+1, len(docs)):
            sim = signature_similarity(perm_sigs[i], perm_sigs[j])
            print(f"D{i} vs D{j}: {sim:.3f}")
    print()

    # --- MinHash with Hash Functions ---
    # Option 1: use random-generated hash functions
    # hash_sigs, all_shingles_hash = minhash_hashing(shingle_sets, num_hashes=10)
    
    # Option 2 (given in question): uncomment below to use given hash functions
    N = len(set().union(*shingle_sets))
    given_funcs = [
         lambda x, n=N: (x + 2) % n,
         lambda x, n=N: (3*x + 1) % n,
         lambda x, n=N: (x + 4) % n
    ]
    hash_sigs, all_shingles_hash = minhash_hashing(shingle_sets, given_funcs=given_funcs)

    print("=== MinHash with Hash Functions ===")
    print("All unique shingles:", all_shingles_hash, "\n")
    for i, sig in enumerate(hash_sigs):
        print(f"D{i} signature: {sig}")
    print()
    for i in range(len(docs)):
        for j in range(i+1, len(docs)):
            sim = signature_similarity(hash_sigs[i], hash_sigs[j])
            print(f"D{i} vs D{j}: {sim:.3f}")



# ==========================================================================================

"""
PageRank using Power Iteration Method
"""

def pagerank(graph, damping_factor=0.85, max_iterations=100, tolerance=1e-6):
    nodes = list(graph.keys())
    n = len(nodes)
    if n == 0:
        return {}

    # Initialize PageRank scores
    pr = {node: 1/n for node in nodes}
    print(f"Initial PageRank: {pr}\n")

    for it in range(1, max_iterations + 1):
        temp_pr = {}
        for node in nodes:
            # Sum PageRank of incoming nodes
            incoming_pr = 0
            for other_node in nodes:
                if node in graph.get(other_node, []):  # if other_node links to node
                    outlinks = len(graph[other_node]) if graph[other_node] else 1
                    incoming_pr += pr[other_node] / outlinks

            # Apply damping factor
            temp_pr[node] = (1 - damping_factor) / n + damping_factor * incoming_pr

        # Print iteration results
        print(f"Iteration {it}: {temp_pr}")

        # Convergence check
        total_diff = sum(abs(temp_pr[node] - pr[node]) for node in nodes)
        pr = temp_pr.copy()
        if total_diff < tolerance:
            print("\nConverged!")
            break

    return pr


# Example usage
if __name__ == "__main__":
    example_graph = {
        'A': ['B'],
        'B': ['A', 'C'],
        'C': ['B'],
    }

    ranks = pagerank(example_graph, damping_factor=0.85, max_iterations=120)
    print("\nFinal PageRank Scores (sorted):")
    for node, rank in sorted(ranks.items(), key=lambda x: x[1], reverse=True):
        print(f"Node {node}: {rank:.4f}")


# ==========================================================================================


#eigen vector page rank
import numpy as np

# Example graph: A->B,C; B->C; C->A; D->C
nodes = ['A', 'B', 'C']
n = len(nodes)
transition = np.zeros((n, n))

# Fill transition matrix (row: from, col: to)
links = {'A': ['B'], 'B': ['A','C'], 'C': ['B']}
node_idx = {node: i for i, node in enumerate(nodes)}

for from_node, to_nodes in links.items():
    out_degree = len(to_nodes)
    for to_node in to_nodes:
        transition[node_idx[from_node], node_idx[to_node]] = 1 / out_degree

# Google matrix with damping=0.85
damping = 0.5
google_matrix = damping * transition + (1 - damping) / n * np.ones((n, n))

# Find eigenvectors; take the one with eigenvalue closest to 1
eigenvalues, eigenvectors = np.linalg.eig(google_matrix.T)  # Transpose for right eigenvector
print(eigenvalues)
print(eigenvectors)
idx = np.argmin(abs(eigenvalues - 1))
print(idx)
pr_vector = abs(eigenvectors[:, idx])
print(pr_vector)
pr_vector /= pr_vector.sum()  # Normalize
print(pr_vector)

# Output
ranks = {nodes[i]: pr_vector[i] for i in range(n)}
print(ranks)