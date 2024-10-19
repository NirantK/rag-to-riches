from itertools import product
from typing import List, Literal, Optional, Union

import pandas as pd
import streamlit as st
from pydantic import BaseModel

vast_gpu_pricing = {
    "RTX 6000ADA": {"min": 0.80, "med": 0.83, "max": 0.93},
    "A100 SXM4": {"min": 0.65, "med": 1.53, "max": 1.75},
    "L40S": {"min": 0.80, "med": 0.87, "max": 0.87},
    "RTX 4090": {"min": 0.29, "med": 0.45, "max": 0.93},
    "H100 SXM": {"min": 2.13, "med": 2.53, "max": 2.67},
    "A100 PCIE": {"min": 0.81, "med": 1.07, "max": 1.07},
    "H100 NVL": {"min": 2.53, "med": 2.80, "max": 3.47},
    "RTX A6000": {"min": 0.45, "med": 0.51, "max": 0.65},
    "RTX 4070S Ti": {"min": 0.20, "med": 0.20, "max": 0.20},
    "H100 PCIE": {"min": 2.53, "med": 3.47, "max": 3.47},
    "L40": {"min": 0.80, "med": 0.80, "max": 0.80},
    "RTX 3080": {"min": 0.08, "med": 0.13, "max": 0.13},
    "A40": {"min": 0.52, "med": 0.67, "max": 0.67},
    "RTX 4070": {"min": 0.16, "med": 0.16, "max": 0.16},
}


gpu_preferences = Literal[
    "RTX 6000ADA",
    "A100 SXM4",
    "L40S",
    "RTX 4090",
    "H100 SXM",
    "A100 PCIE",
    "H100 NVL",
    "RTX A6000",
    "RTX 4070S Ti",
    "H100 PCIE",
    "L40",
    "RTX 3080",
    "A40",
    "RTX 4070",
    "Cloud",
    "TogetherAI",
]


class RAGComponent(BaseModel):
    name: str
    memory_usage_in_gb: Union[float, None]
    disk_space_in_gb: Union[float, None]
    gpu_type: Union[gpu_preferences, None]
    gpu_count: int = 0
    latency_ms: Union[float, None] = None
    throughput_tokens_per_second: Union[float, None] = None
    cost_per_million_tokens: Union[float, None] = None
    cost_per_search: Union[float, None] = None
    component_type: Literal["LLM", "Embedding Model", "Reranker Model", "Vector Store"]


llms = [
    RAGComponent(
        name="gpt-4o",
        memory_usage_in_gb=None,
        disk_space_in_gb=None,
        gpu_type="Cloud",
        gpu_count=0,
        latency_ms=None,
        throughput_tokens_per_second=None,
        cost_per_million_tokens=0.50,
        component_type="LLM",
    ),
    RAGComponent(
        name="gpt-4o-mini",
        memory_usage_in_gb=None,
        disk_space_in_gb=None,
        gpu_type="Cloud",
        gpu_count=0,
        latency_ms=None,
        throughput_tokens_per_second=None,
        cost_per_million_tokens=0.06,
        component_type="LLM",
    ),
    RAGComponent(
        name="llama-3b",
        memory_usage_in_gb=None,
        disk_space_in_gb=None,
        gpu_type="TogetherAI",
        gpu_count=0,
        latency_ms=None,
        throughput_tokens_per_second=None,
        cost_per_million_tokens=0.06,
        component_type="LLM",
    ),
    RAGComponent(
        name="llama-8b",
        memory_usage_in_gb=None,
        disk_space_in_gb=None,
        gpu_type="TogetherAI",
        gpu_count=0,
        latency_ms=None,
        throughput_tokens_per_second=None,
        cost_per_million_tokens=0.20,
        component_type="LLM",
    ),
    RAGComponent(
        name="llama-11b",
        memory_usage_in_gb=None,
        disk_space_in_gb=None,
        gpu_type="TogetherAI",
        gpu_count=0,
        latency_ms=None,
        throughput_tokens_per_second=None,
        cost_per_million_tokens=0.18,
        component_type="LLM",
    ),
    RAGComponent(
        name="llama-70b",
        memory_usage_in_gb=None,
        disk_space_in_gb=None,
        gpu_type="TogetherAI",
        gpu_count=0,
        latency_ms=None,
        throughput_tokens_per_second=None,
        cost_per_million_tokens=0.9,
        component_type="LLM",
    ),
    RAGComponent(
        name="llama-90b-vision",
        memory_usage_in_gb=None,
        disk_space_in_gb=None,
        gpu_type="TogetherAI",
        gpu_count=0,
        latency_ms=None,
        throughput_tokens_per_second=None,
        cost_per_million_tokens=1.20,
        component_type="LLM",
    ),
    RAGComponent(
        name="llama-405b",
        memory_usage_in_gb=None,
        disk_space_in_gb=None,
        gpu_type="Cloud",
        gpu_count=0,
        latency_ms=None,
        throughput_tokens_per_second=None,
        cost_per_million_tokens=3.50,
        component_type="LLM",
    ),
]

embedding_models = [
    RAGComponent(
        name="Cohere-Embed-v3",
        memory_usage_in_gb=None,
        disk_space_in_gb=None,
        gpu_type="Cloud",
        gpu_count=0,
        latency_ms=100,
        throughput_tokens_per_second=None,
        cost_per_million_tokens=0.1,
        component_type="Embedding Model",
    ),
    RAGComponent(
        name="all-MiniLM-L6-v2",
        memory_usage_in_gb=None,
        disk_space_in_gb=None,
        gpu_type=None,
        gpu_count=0,
        latency_ms=100,
        throughput_tokens_per_second=None,
        cost_per_million_tokens=0.05,
        component_type="Embedding Model",
    ),
]

reranker_models = [
    RAGComponent(
        name="Cohere-Rerank-v3",
        memory_usage_in_gb=None,
        disk_space_in_gb=None,
        gpu_type="Cloud",
        gpu_count=0,
        latency_ms=None,
        throughput_tokens_per_second=None,
        cost_per_million_tokens=None,
        cost_per_search=0.002,
        component_type="Reranker Model",
    ),
]

vector_stores = [
    RAGComponent(
        name="Self Hosted Qdrant",
        memory_usage_in_gb=None,
        disk_space_in_gb=None,
        gpu_type=None,
        gpu_count=0,
        latency_ms=20,
        throughput_tokens_per_second=None,
        cost_per_million_tokens=None,
        component_type="Vector Store",
    ),
]


def get_gpu_pricing(gpu_preference: gpu_preferences):
    if gpu_preference in vast_gpu_pricing.keys():
        return vast_gpu_pricing[gpu_preference]["med"]
    else:
        return 0.0


def generate_model_list_combinations(
    llms, embedding_models, reranker_models, vector_stores
):
    """
    Generates all possible combinations of models from the provided list.
    """
    import random

    return [
        random.choice(llms),
        random.choice(embedding_models),
        random.choice(reranker_models),
        random.choice(vector_stores),
    ]


def create_model_row(component, name, model):
    return {
        "Component": component,
        "Name": name,
        "GPU Type": model.gpu_type,
        "GPU Count": model.gpu_count,
        "Cost per Million Tokens": f"${model.cost_per_million_tokens:.4f}"
        if model.cost_per_million_tokens
        else "N/A",
    }


def display_selected_models(
    selected_llm,
    selected_embedding,
    selected_reranker,
    selected_vector_store,
    selected_models,
):
    # Create a DataFrame to display the selected models
    selected_models_df = pd.DataFrame(
        [
            create_model_row("LLM", selected_llm, selected_models[0]),
            create_model_row("Embedding Model", selected_embedding, selected_models[1]),
            create_model_row("Reranker Model", selected_reranker, selected_models[2]),
            create_model_row("Vector Store", selected_vector_store, selected_models[3]),
        ]
    )

    # Display the DataFrame as a table
    st.table(selected_models_df)


def vector_store_costs(
    num_questions: int, 
    num_tokens: int,
    num_dimensions: int,
    chunk_size: int,
    overlap: int,
    move_index_disk: bool,
    move_vectors_disk: bool,
    binary_quantization: bool,
    product_quantization: bool,
):
    # Base latency and throughput
    latency = 100  # in ms, base value
    throughput = 10  # in queries per second, base value
    # Estimate number of vectors
    effective_chunk_size = chunk_size - overlap
    num_vectors = max(
        1, (num_tokens * 10**6 - overlap) / effective_chunk_size
    )  # Accounting for overlap in rolling window
    vcpu_count = num_questions // 64
    vcpu_count = max(vcpu_count, 1)
    instance_cost = vcpu_count * 0.09336  # $0.09336 per vCPU per hour
    # Estimate RAM usage
    # Assume 4 bytes per dimension for vector embeddings, with 768 dimensions as a common size
    vector_size_bytes = num_dimensions * 4
    vector_space = (num_vectors * vector_size_bytes) * (1/2**30)  # vectors in GB
    index_space = vector_space * 0.5  # 50% overhead for the index
    # Each token is 1 byte
    token_space = (num_tokens * 1 * 10**6) * (1/2**30)  # tokens in GB

    # Default, everything in RAM
    ram_usage = vector_space + index_space + token_space
    disk_space = 0

    # Calculate the effects of all optimizations combined
    ram_reduction = 0
    disk_increase = 0
    latency_multiplier = 1
    throughput_multiplier = 1

    if move_index_disk:
        ram_reduction += index_space
        disk_increase += index_space
        latency_multiplier *= 1.1
        throughput_multiplier *= 0.9

    if move_vectors_disk:
        ram_reduction += vector_space
        disk_increase += vector_space
        latency_multiplier *= 1.2
        throughput_multiplier *= 0.8

    if binary_quantization:
        # BQ means full vectors are moved to disk, binary vectors and index in RAM
        ram_reduction += (vector_space*(1/30))
        disk_increase += vector_space
        latency_multiplier *= 1.2
        throughput_multiplier *= 0.9

    if product_quantization:
        # PQ means full vectors are moved to disk, half the vectors are stored in RAM and all of the index
        ram_reduction += vector_space * 0.5
        disk_increase += vector_space * 0.5
        latency_multiplier *= 1.1
        throughput_multiplier *= 0.8

    # Apply the combined effects
    ram_usage -= ram_reduction
    disk_space += disk_increase
    latency *= latency_multiplier
    throughput *= throughput_multiplier

    # Calculate costs
    compute_cost = instance_cost * 24 * 30
    storage_cost = disk_space * 0.05  # Assuming $0.05 per GB of storage per month
    # Assuming $0.072 per GB of RAM per day, 24 hours per day, 30 days per month
    memory_cost = ram_usage * 0.072 * 24 * 30
    # Calculate total cost
    total_cost = compute_cost + storage_cost + memory_cost

    # Round the results for better readability
    compute_cost = round(compute_cost, 2)
    storage_cost = round(storage_cost, 2)
    memory_cost = round(memory_cost, 2)
    total_cost = round(total_cost, 2)

    # Return all calculated values including costs
    return {
        "Number of Vectors": num_vectors,
        "Latency (ms)": latency,
        "Throughput (queries/sec)": throughput,
        "RAM Usage (GB)": ram_usage,
        "Disk Space (GB)": disk_space,
        "Instance Cost (USD)": instance_cost,
        "Storage Cost (USD)": storage_cost,
        "Memory Cost (USD)": memory_cost,
        "Estimated Monthly Operational Cost (USD)": total_cost,
    }


def llm_costs(
    input_tokens: int, output_tokens: int, num_questions: int, model: RAGComponent
):
    total_tokens = (input_tokens + output_tokens) * num_questions
    if model.gpu_type == "Cloud" or model.gpu_type == "TogetherAI":
        cost_per_million_tokens = model.cost_per_million_tokens
    return {
        "Number of Questions": num_questions,
        "Input Tokens": input_tokens * num_questions,
        "Output Tokens": output_tokens * num_questions,
        "Total Tokens": total_tokens,
        "Cost per Million Input Tokens": cost_per_million_tokens,
        "Cost per Million Output Tokens": cost_per_million_tokens,
        "Total Cost": total_tokens * cost_per_million_tokens / 1000000,
    }


def reranker_costs(num_questions: int, model: RAGComponent):
    return {
        "Number of Questions": num_questions,
        "Cost per Search": model.cost_per_search,
        "Total Cost": num_questions * model.cost_per_search,
    }


def embedding_costs(num_tokens: int, num_dimensions: int, model: RAGComponent):
    return {
        "Number of Tokens": num_tokens,
        "Number of Dimensions": num_dimensions,
        "Cost per Million Tokens": model.cost_per_million_tokens,
    }


def main():
    st.title("Cost and Performance Analysis Simulator")

    st.sidebar.header("Simulation Parameters")

    # Cost Analysis Parameters
    st.sidebar.subheader("Cost Analysis for 100M Token Search")
    num_tokens = st.sidebar.number_input(
        "Number of Tokens (in millions)", value=1, step=1, min_value=0
    )
    num_questions = st.sidebar.number_input(
        "Number of Questions", value=1, step=1, min_value=0
    )

    # Chunk Size and Overlap
    st.sidebar.subheader("Chunking Parameters")
    chunk_size = st.sidebar.slider(
        "Chunk Size (tokens)",
        min_value=128,
        max_value=2048,
        value=512,
        step=128,
        key="chunk_size",
    )
    overlap_percentage = st.sidebar.slider(
        "Overlap (%)",
        min_value=0,
        max_value=100,
        value=25,
        step=5,
        key="overlap_percentage",
    )
    overlap = int(chunk_size * overlap_percentage / 100)
    num_dimensions = st.sidebar.slider(
        "Number of Dimensions",
        min_value=96,
        max_value=4096,
        value=768,
        step=128,
        key="num_dimensions",
    )

    # Prompt Tokens
    st.sidebar.subheader("Input Tokens")
    input_tokens = st.sidebar.number_input(
        "Input Tokens", value=10, step=1, min_value=0
    )
    output_tokens = st.sidebar.number_input(
        "Output Tokens", value=10, step=1, min_value=0
    )

    # Optimization Options
    st.sidebar.subheader("Optimization Options")
    move_index_disk = st.sidebar.checkbox("Move Index to Disk", value=False)
    move_vectors_disk = st.sidebar.checkbox("Move Vectors to Disk", value=False)
    binary_quantization = st.sidebar.checkbox("Binary Quantization", value=False)
    product_quantization = st.sidebar.checkbox("Product Quantization", value=False)

    # Add dropdown lists in the sidebar for model selection
    st.sidebar.subheader("Model Selection")
    selected_llm = st.sidebar.selectbox(
        "Select LLM", options=[model.name for model in llms], index=0
    )
    selected_embedding = st.sidebar.selectbox(
        "Select Embedding Model",
        options=[model.name for model in embedding_models],
        index=0,
    )
    selected_reranker = st.sidebar.selectbox(
        "Select Reranker Model",
        options=[model.name for model in reranker_models],
        index=0,
    )
    selected_vector_store = st.sidebar.selectbox(
        "Select Vector Store", options=[model.name for model in vector_stores], index=0
    )

    # Get the selected models
    selected_models = [
        next(model for model in llms if model.name == selected_llm),
        next(model for model in embedding_models if model.name == selected_embedding),
        next(model for model in reranker_models if model.name == selected_reranker),
        next(model for model in vector_stores if model.name == selected_vector_store),
    ]

    st.subheader("Selected Model Combination")
    display_selected_models(
        selected_llm,
        selected_embedding,
        selected_reranker,
        selected_vector_store,
        selected_models,
    )

    st.title("Operational Considerations")
    for component in selected_models:
        if component.component_type == "LLM":
            st.subheader(f"{component.component_type} {component.name} Costs")
            component_cost = llm_costs(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                num_questions=num_questions,
                model=component,
            )
            st.table(component_cost)
        if component.component_type == "Vector Store":
            st.subheader(f"{component.component_type} {component.name} Costs")
            component_cost = vector_store_costs(
                num_questions=num_questions,
                num_tokens=num_tokens,
                num_dimensions=num_dimensions,
                chunk_size=chunk_size,
                overlap=overlap,
                move_index_disk=move_index_disk,
                move_vectors_disk=move_vectors_disk,
                binary_quantization=binary_quantization,
                product_quantization=product_quantization,
            )
            st.table(component_cost)
        if component.component_type == "Embedding Model":
            st.subheader(f"{component.component_type} {component.name} Costs")
            component_cost = embedding_costs(
                num_tokens=num_tokens, num_dimensions=num_dimensions, model=component
            )
            st.table(component_cost)
        if component.component_type == "Reranker Model":
            st.subheader(f"{component.component_type} {component.name} Costs")
            component_cost = reranker_costs(
                num_questions=num_questions, model=component
            )
            st.table(component_cost)


if __name__ == "__main__":
    main()
