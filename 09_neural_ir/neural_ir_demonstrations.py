import json
import random

import numpy as np
from fastembed import (
    LateInteractionTextEmbedding,
    SparseTextEmbedding,
    TextEmbedding,
)
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sentence_transformers import CrossEncoder
from tokenizers import Tokenizer

random.seed(42)
# Initialize console for rich output
console = Console()

# List supported models for SparseTextEmbedding
SparseTextEmbedding.list_supported_models()
# Sample data
documents = [
    "Definitely",
    "Hell yeah",
    "Perhaps, I'll think about it",
    "Maybe",
    "I am not sure about that",
    "Hmm, no?",
    "No way on earth",
    "I'll pass",
]

queries = ["Yes", "Maybe", "No"]
# Initialize models
model_name = "prithivida/Splade_PP_en_v1"
sparse_model = SparseTextEmbedding(model_name=model_name, device="mps")
dense_embedding_model = TextEmbedding(device="mps")
late_interaction_model = LateInteractionTextEmbedding(
    "colbert-ir/colbertv2.0", device="mps"
)
cross_encoder_model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-12-v2", max_length=512
)


# Generate embeddings for different models
def generate_embeddings(model, data, batch_size=32, query=False):
    if query:
        return list(model.query_embed(data, batch_size=batch_size))
    return list(model.passage_embed(data, batch_size=batch_size))


# ColBERT embeddings
colbert_document_embeddings = generate_embeddings(late_interaction_model, documents)
colbert_query_embeddings = generate_embeddings(
    late_interaction_model, queries, query=True
)

# Dense embeddings
dense_document_embeddings = generate_embeddings(dense_embedding_model, documents)
dense_query_embeddings = generate_embeddings(dense_embedding_model, queries, query=True)

# Sparse embeddings
sparse_document_embeddings = generate_embeddings(sparse_model, documents)
sparse_query_embeddings = generate_embeddings(sparse_model, queries, query=True)


# ColBERT-specific functions
colbert_tokenizer = Tokenizer.from_pretrained("colbert-ir/colbertv2.0")


# Print embedding shapes/lengths
def print_embedding_info(name, doc_emb, query_emb):
    console.print(
        f"{name} Document Embedding Shape/Length: {doc_emb[0].shape if hasattr(doc_emb[0], 'shape') else len(doc_emb[0].indices)}"
    )
    console.print(
        f"{name} Query Embedding Shape/Length: {query_emb[0].shape if hasattr(query_emb[0], 'shape') else len(query_emb[0].indices)}"
    )


def get_sparse_tokens_and_weights(sparse_embedding, tokenizer):
    token_weight_dict = {}
    for i in range(len(sparse_embedding.indices)):
        token = tokenizer.decode([sparse_embedding.indices[i]])
        weight = sparse_embedding.values[i]
        token_weight_dict[token] = weight

    # Sort the dictionary by weights
    token_weight_dict = dict(
        sorted(token_weight_dict.items(), key=lambda item: item[1], reverse=True)
    )
    return token_weight_dict


def display_token_representations(
    documents: list[str] = documents,
    queries: list[str] = queries,
    display_queries: bool = True,
    display_documents: bool = True,
):
    sparse_tokenizer = Tokenizer.from_pretrained(
        SparseTextEmbedding.list_supported_models()[0]["sources"]["hf"]
    )
    if display_documents:
        for doc_index, doc in enumerate(documents):
            console.print(
                Panel(
                    f"[cyan]Document:[/cyan] {documents[doc_index]}\n\n"
                    f"[magenta]Sparse Document Representation:[/magenta]\n"
                    f"{json.dumps(get_sparse_tokens_and_weights(sparse_document_embeddings[doc_index], sparse_tokenizer), indent=2)}\n\n"
                    f"[green]Dense Document Representation:[/green] Array of length {len(dense_document_embeddings[doc_index])}\n\n"
                    f"[blue]ColBERT Document Representation:[/blue] Tensor of shape {colbert_document_embeddings[doc_index].shape}",
                    title="Document Representations",
                )
            )

    if display_queries:
        for query_index, query in enumerate(queries):
            console.print(
                Panel(
                    f"[cyan]Query:[/cyan] {query}\n\n"
                    f"[magenta]Sparse Query Representation:[/magenta]\n"
                    f"{json.dumps(get_sparse_tokens_and_weights(sparse_query_embeddings[query_index], sparse_tokenizer), indent=2)}\n\n"
                    f"[green]Dense Query Representation:[/green] Array of length {len(dense_query_embeddings[query_index])}\n\n"
                    f"[blue]ColBERT Query Representation:[/blue] Tensor of shape {colbert_query_embeddings[query_index].shape}",
                    title="Query Representations",
                    border_style="bold",
                )
            )


def compute_relevance_scores(query_embedding: np.array, document_embeddings: np.array):
    # Compute batch dot-product of query_embedding and document_embeddings
    # Resulting shape: [num_documents, num_query_terms, max_doc_length]
    return np.matmul(query_embedding, document_embeddings.transpose(0, 2, 1))


class ColBERTInteraction:
    def __init__(self, query_index, doc_index):
        self.query_index = query_index
        self.doc_index = doc_index

    def display_interaction(
        self,
        display_token_scores: bool = True,
        display_max_scores: bool = True,
        display_total_score: bool = True,
    ):
        query = queries[self.query_index]
        query_embedding = colbert_query_embeddings[self.query_index]
        document_embedding = np.array(colbert_document_embeddings)
        scores = compute_relevance_scores(query_embedding, document_embedding)
        if display_token_scores:
            console.print(f"Query Embedding Shape: {query_embedding.shape}")
            console.print(f"Document Embedding Shape: {document_embedding.shape}")
            console.print(f"Scores Shape: {scores.shape}")

        # Get the tokenizer for ColBERT
        colbert_tokenizer = Tokenizer.from_pretrained("colbert-ir/colbertv2.0")
        document = documents[self.doc_index]
        # Tokenize the query and document
        query_token_ids = colbert_tokenizer.encode(query).ids
        document_tokens_ids = colbert_tokenizer.encode(document).ids
        string_query_tokens = [
            colbert_tokenizer.decode([token_id]) for token_id in query_token_ids
        ]
        string_document_tokens = [
            colbert_tokenizer.decode([token_id]) for token_id in document_tokens_ids
        ]
        if display_token_scores:
            console.print(f"Query Tokens: {string_query_tokens}")
            console.print(f"Document Tokens: {string_document_tokens}")

        # Create a mapping of scores to query, document token pair
        score_token_mapping = []
        for i, q_token in enumerate(string_query_tokens):
            for j, d_token in enumerate(string_document_tokens):
                score_token_mapping.append(
                    {
                        "query_token": q_token,
                        "document_token": d_token,
                        "score": scores[
                            self.doc_index, i, j
                        ],  # Select the score based on document_idx and assume all queries are padded to 32
                    }
                )

        # console.print(f"Score Token Mapping: {score_token_mapping}")
        # Print as table
        if display_token_scores:
            score_table = Table(title="Score Token Mapping")
            score_table.add_column("Query Token", style="cyan")
            score_table.add_column("Document Token", style="magenta")
            score_table.add_column("Score", style="green")
            for mapping in score_token_mapping:
                score_table.add_row(
                    mapping["query_token"],
                    mapping["document_token"],
                    f"{mapping['score']:.2f}",
                )
            console.print(score_table)

        max_scores_per_query_term = np.max(scores, axis=2)

        if display_max_scores:
            # Create a table for max scores per query term
            max_score_table = Table(title="Max Scores per Query Term")
            max_score_table.add_column("Query Term", style="cyan")
            max_score_table.add_column("Document", style="magenta")
            max_score_table.add_column("Max Score", style="green")

            for i, q_token in enumerate(string_query_tokens):
                max_score = max_scores_per_query_term[self.doc_index, i]
                max_score_table.add_row(
                    q_token, documents[self.doc_index], f"{max_score:.2f}"
                )

            console.print(max_score_table)
        total_score = np.sum(scores[self.doc_index])
        if display_total_score:
            # Create a panel for the total score
            total_score_panel = Panel(
                f"Query: {query}\nDocument: {documents[self.doc_index]}\nTotal Score: {total_score:.4f}",
                title="ColBERT Total Score",
                expand=False,
                border_style="green",
            )
            console.print(total_score_panel)

        return total_score


# Demonstrate Cross-Encoder
def demonstrate_cross_encoder(queries, documents):
    # Use the existing documents and queries from the file
    queries = queries
    documents = documents
    scores = cross_encoder_model.predict([(q, d) for q in queries for d in documents])

    score_table = Table(title="Cross-Encoder Scores")
    score_table.add_column("Query/Document", style="cyan")
    for doc in documents:
        score_table.add_column(doc, style="magenta")

    for i, query in enumerate(queries):
        row = [query]
        for j, _ in enumerate(documents):
            score_index = i * len(documents) + j
            row.append(f"{scores[score_index]:.4f}")
        score_table.add_row(*row)

    console.print(score_table)


# print_embedding_info("ColBERT", colbert_document_embeddings, colbert_query_embeddings)
# print_embedding_info("Dense", dense_document_embeddings, dense_query_embeddings)
# print_embedding_info("Sparse", sparse_document_embeddings, sparse_query_embeddings)


########################################################
# Dense Interaction
########################################################

total_scores = []
for i in range(len(documents)):
    for j in range(len(queries)):
        similarity_score = np.dot(
            dense_query_embeddings[j], dense_document_embeddings[i]
        )
        total_scores.append(
            {
                "document": documents[i],
                "query": queries[j],
                "total_score": similarity_score,
            }
        )

# Sort total_scores by 'total_score' in descending order
sorted_scores = sorted(total_scores, key=lambda x: x["total_score"], reverse=True)

# Create a table to display the sorted scores
score_table = Table(title="Sorted Dense Total Scores")
score_table.add_column("Document", style="cyan")
score_table.add_column("Query", style="magenta")
score_table.add_column("Total Score", style="green")
# Add rows to the table
for score in sorted_scores:
    score_table.add_row(
        score["document"], score["query"], f"{score['total_score']:.4f}"
    )

# Print the table
console.print(score_table)

########################################################
# ColBERT Interaction
########################################################

total_scores = []
for i in range(len(documents)):
    for j in range(len(queries)):
        late_interaction = ColBERTInteraction(doc_index=i, query_index=j)
        total_score = late_interaction.display_interaction(
            display_token_scores=False,
            display_max_scores=False,
            display_total_score=False,
        )
        total_scores.append(
            {"document": documents[i], "query": queries[j], "total_score": total_score}
        )

# Sort total_scores by 'total_score' in descending order
sorted_scores = sorted(total_scores, key=lambda x: x["total_score"], reverse=True)

# Create a table to display the sorted scores
score_table = Table(title="Sorted Colbert Total Scores")
score_table.add_column("Document", style="cyan")
score_table.add_column("Query", style="magenta")
score_table.add_column("Total Score", style="green")

# Add rows to the table
for score in sorted_scores:
    score_table.add_row(
        score["document"], score["query"], f"{score['total_score']:.4f}"
    )

# Print the table
console.print(score_table)

########################################################
# Sparse Interaction
########################################################

# display_token_representations(documents=documents, queries=queries)

total_scores = []
for i in range(len(documents)):
    for j in range(len(queries)):
        query_indices, query_weights = (
            sparse_document_embeddings[i].indices,
            sparse_document_embeddings[i].values,
        )
        document_indices, document_weights = (
            sparse_query_embeddings[j].indices,
            sparse_query_embeddings[j].values,
        )

        # Find all indices that are in both query_indices and document_indices
        common_indices = set(query_indices) & set(document_indices)
        similarity_score = 0
        for index in common_indices:
            # Get the weight of the token in the query and document using lookup and then multiply them
            query_weight = query_weights[np.where(query_indices == index)[0][0]]
            document_weight = document_weights[
                np.where(document_indices == index)[0][0]
            ]
            similarity_score += query_weight * document_weight

        total_scores.append(
            {
                "document": documents[i],
                "query": queries[j],
                "total_score": similarity_score,
            }
        )

# Sort total_scores by 'total_score' in descending order
sorted_scores = sorted(total_scores, key=lambda x: x["total_score"], reverse=True)

# Create a table to display the sorted scores
score_table = Table(title="Sorted Sparse Total Scores")
score_table.add_column("Document", style="cyan")
score_table.add_column("Query", style="magenta")
score_table.add_column("Total Score", style="green")

# Add rows to the table
for score in sorted_scores:
    score_table.add_row(
        score["document"], score["query"], f"{score['total_score']:.4f}"
    )

# Print the table
console.print(score_table)
