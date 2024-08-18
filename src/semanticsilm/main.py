import os
from llama_index.core import SimpleDirectoryReader, KnowledgeGraphIndex, StorageContext
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai.utils import CompletionResponse
from llama_index.core import Settings
from llama_index.core import load_index_from_storage
from thefuzz import fuzz
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PERSIST_DIR = "./persist"

IMPORTANT_ENTITIES = [
    "Ilúvatar", "Valar", "Maiar", "Elves", "Men", "Dwarves", "Melkor", "Morgoth", "Fëanor",
    "Valinor", "Middle-earth", "Beleriand", "Númenor", "Arda",
    "Silmarils", "Rings of Power", "Eru", "Manwë", "Varda", "Ulmo", "Aulë", "Yavanna",
    "Mandos", "Lórien", "Tulkas", "Oromë", "Nessa", "Nienna", "Estë", "Vairë", "Vána",
    "Finwë", "Indis", "Míriel", "Fingolfin", "Finarfin", "Galadriel", "Celeborn",
    "Thingol", "Melian", "Lúthien", "Beren", "Túrin", "Húrin", "Morwen", "Eärendil", "Elwing",
    "Eönwë", "Sauron", "Gothmog", "Ungoliant", "Glaurung", "Ancalagon"
]

IMPORTANT_RELATIONSHIPS = [
    "created", "ruled", "fought against", "allied with", "descended from",
    "possessed", "destroyed", "resided in", "journeyed to", "married",
    "betrayed", "swore oath to", "crafted", "guarded", "taught", "learned from",
    "imprisoned", "rescued", "cursed", "blessed", "counseled", "served",
    "challenged", "defeated", "fled from", "pursued", "sang", "shaped", "corrupted"
]

aliases = {
    "morgoth": "melkor",
    "gorthaur": "sauron",
    "mithrandir": "gandalf",
    "olórin": "gandalf",
    "elessar": "aragorn",
    "elfstone": "aragorn",
    "tharkûn": "gandalf",
    "incánus": "gandalf",
}

def silmarillion_triplet_extract_fn(text):
    prompt = f"""
    Extract key relationships from the following text, focusing on characters, 
    locations, and events from the Silmarillion. Pay special attention to:
    
    Entities: {', '.join(IMPORTANT_ENTITIES)}
    Relationships: {', '.join(IMPORTANT_RELATIONSHIPS)}
    
    Format as (Entity1, Relationship, Entity2).
    Only extract relationships that are explicitly stated or strongly implied.
    Limit to 15 most important relationships.
    
    Text: {text}
    
    Relationships:
    """
    logger.debug(f"Sending prompt to LLM (length: {len(prompt)} chars)")
    response = Settings.llm.complete(prompt)
    logger.debug(f"Received response from LLM (length: {len(response.text)} chars)")
    logger.debug(f"LLM response: {response.text}")
    
    triplets = parse_response_to_triplets(response)
    
    if not triplets:
        logger.warning("No triplets extracted. LLM response may not contain the expected format.")
    
    return triplets

def parse_response_to_triplets(response: CompletionResponse):
    triplets = []
    for line in response.text.split('\n'):
        line = line.strip()
        # remove the number prefix if present
        line = re.sub(r'^\d+\.\s*', '', line)
        if line.startswith('(') and line.endswith(')'):
            parts = line[1:-1].split(',')
            if len(parts) == 3:
                triplets.append(tuple(part.strip() for part in parts))
        elif line:
            logger.debug(f"Non-triplet line in response: {line}")
    
    logger.debug(f"Extracted {len(triplets)} triplets from LLM response")
    for triplet in triplets:
        logger.debug(f"Triplet: {triplet}")
    
    return triplets


def preprocess_entity(entity):
    return entity.lower().strip()

def are_entities_similar(entity1, entity2, threshold=80):
    return fuzz.ratio(preprocess_entity(entity1), preprocess_entity(entity2)) > threshold

def silmarillion_entity_linker(triplets):
    entity_map = {}
    linked_triplets = []
    
    for triplet in triplets:
        linked_triplet = []
        for entity in triplet:
            processed_entity = preprocess_entity(entity)
            matched_entity = None
            for known_entity in entity_map:
                if are_entities_similar(processed_entity, known_entity):
                    matched_entity = entity_map[known_entity]
                    break
            if matched_entity is None:
                matched_entity = entity
                entity_map[processed_entity] = entity
            linked_triplet.append(matched_entity)
        linked_triplets.append(tuple(linked_triplet))
    
    for i, triplet in enumerate(linked_triplets):
        new_triplet = []
        for entity in triplet:
            lower_entity = entity.lower()
            if lower_entity in aliases:
                new_triplet.append(aliases[lower_entity].title())
            else:
                new_triplet.append(entity)
        linked_triplets[i] = tuple(new_triplet)
    
    return linked_triplets

def create_silmarillion_kg(documents):
    graph_store = SimpleGraphStore()
    storage_context = StorageContext.from_defaults(graph_store=graph_store)

    logger.info(f"Creating initial Knowledge Graph Index from {len(documents)} documents...")
    
    all_triplets = []
    for doc in documents:
        logger.info(f"Processing document: {doc.doc_id}")
        text = doc.text
        triplets = silmarillion_triplet_extract_fn(text)
        logger.info(f"Extracted {len(triplets)} triplets from document")
        logger.debug(f"Triplets: {triplets}")
        all_triplets.extend(triplets)
    
    logger.info(f"Total triplets extracted: {len(all_triplets)}")
    
    if not all_triplets:
        logger.warning("No triplets were extracted. The knowledge graph will be empty.")
    
    index = KnowledgeGraphIndex.from_documents(
        documents,
        max_triplets_per_chunk=15,
        kg_triplet_extract_fn=silmarillion_triplet_extract_fn,
        include_embeddings=True,
        storage_context=storage_context,
    )
    
    logger.info("Knowledge Graph Index created. Inspecting graph structure...")
    inspect_graph_structure(index.graph_store)
    
    logger.info("Performing entity linking...")
    linked_graph_store = silmarillion_entity_linking(index.graph_store)
    
    logger.info("Updating index with linked graph store...")
    index._graph_store = linked_graph_store
    index._storage_context.graph_store = linked_graph_store
    
    logger.info("Final Knowledge Graph structure:")
    inspect_graph_structure(index.graph_store)

    return index

def inspect_graph_structure(graph_store: SimpleGraphStore):
    total_subjects = len(graph_store._data.graph_dict)
    logger.info(f"Total subjects in graph: {total_subjects}")
    
    if total_subjects == 0:
        logger.warning("The graph is empty. No entities or relationships were extracted.")
        return
    
    total_relationships = sum(len(relations) for relations in graph_store._data.graph_dict.values())
    logger.info(f"Total relationships in graph: {total_relationships}")
    
    sample_size = min(5, total_subjects)
    logger.debug(f"Sample of graph structure (up to {sample_size} subjects):")
    for subject, relations in list(graph_store._data.graph_dict.items())[:sample_size]:
        logger.debug(f"Subject: {subject}")
        for relation, object in relations[:5]:
            logger.debug(f"  - {relation} -> {object}")
        if len(relations) > 5:
            logger.debug(f"  ... and {len(relations) - 5} more relations")

def silmarillion_entity_linking(graph_store: SimpleGraphStore) -> SimpleGraphStore:
    linked_graph_store = SimpleGraphStore()
    entity_map = {}
    
    logger.info(f"Starting entity linking process. Total subjects: {len(graph_store._data.graph_dict)}")
    for subj, relations in graph_store._data.graph_dict.items():
        linked_subj = link_entity(subj, entity_map)
        logger.debug(f"Linking subject: {subj} -> {linked_subj}")
        for rel, obj in relations:
            linked_obj = link_entity(obj, entity_map)
            logger.debug(f"Linking object: {obj} -> {linked_obj}")
            aliased_subj = apply_alias(linked_subj)
            aliased_obj = apply_alias(linked_obj)
            logger.debug(f"Adding triplet: ({aliased_subj}, {rel}, {aliased_obj})")
            linked_graph_store.upsert_triplet(aliased_subj, rel, aliased_obj)
    
    logger.info(f"Entity linking complete. Total linked subjects: {len(linked_graph_store._data.graph_dict)}")
    return linked_graph_store

def apply_alias(entity: str) -> str:
    lower_entity = entity.lower()
    if lower_entity in aliases:
        aliased = aliases[lower_entity].title()
        logger.info(f"Applied alias: {entity} -> {aliased}")
        return aliased
    return entity

def link_entity(entity: str, entity_map: dict) -> str:
    processed_entity = preprocess_entity(entity)
    for known_entity, mapped_entity in entity_map.items():
        if are_entities_similar(processed_entity, known_entity):
            logger.info(f"Linked entity: {entity} -> {mapped_entity} (via {known_entity})")
            return mapped_entity
    entity_map[processed_entity] = entity
    logger.info(f"New entity encountered: {entity}")
    return entity

def main():
    documents = SimpleDirectoryReader('./data', recursive=True, filename_as_id=True).load_data()
    print("Loaded docs")

    llm = OpenAI(temperature=0, model="gpt-4o-mini")
    Settings.llm = llm
    Settings.chunk_size = 1024

    if not os.path.exists(PERSIST_DIR):
        index = create_silmarillion_kg(documents)
        storage_context = index.storage_context
        storage_context.persist(persist_dir=PERSIST_DIR)
        print("Built and saved index")
    else:
        storage_context = StorageContext.from_defaults(
            graph_store=SimpleGraphStore.from_persist_dir(PERSIST_DIR),
            index_store=SimpleIndexStore.from_persist_dir(PERSIST_DIR)
        )
        index = load_index_from_storage(storage_context)
        print("Loaded index")

    query_engine = index.as_query_engine(include_text=True, response_mode="tree_summarize")

    while True:
        query = input("Enter query (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
        response = query_engine.query(query)
        print(response)

if __name__ == "__main__":
    main()