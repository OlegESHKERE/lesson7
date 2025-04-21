import os
import json
import httpx
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

# Налаштування
load_dotenv()
COLLECTION_NAME = "products"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Ініціалізація Qdrant
client = QdrantClient(location=":memory:")

# Завантаження даних
with open("products.json", "r", encoding="utf-8") as f:
    products = json.load(f)

# Ініціалізація моделі для векторного пошуку
encoder = SentenceTransformer(EMBEDDING_MODEL)

# Ініціалізація колекції
try:
    client.get_collection(COLLECTION_NAME)
    client.delete_collection(COLLECTION_NAME)
except Exception:
    pass

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=encoder.get_sentence_embedding_dimension(),
        distance=Distance.COSINE
    )
)

# Додавання даних
points = []
for idx, product in enumerate(products):
    text = f"{product['name']} {product['description']}"
    vector = encoder.encode(text).tolist()
    
    points.append(
        PointStruct(
            id=idx,
            vector=vector,
            payload={
                "name": product["name"],
                "price": product["price"],
                "brand": product["brand"],
                "category": product["category"]
            }
        )
    )

client.upsert(collection_name=COLLECTION_NAME, points=points)

### Векторний пошук ###
def search_products(query: str, top_k: int = 3, brand_filter: str = None):
    query_vector = encoder.encode(query).tolist()
    
    qdrant_filter = None
    if brand_filter:
        qdrant_filter = Filter(
            must=[FieldCondition(key="brand", match=MatchValue(value=brand_filter))]
        )
    
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        query_filter=qdrant_filter,
        limit=top_k
    )
    
    return [{
        "name": hit.payload["name"],
        "price": hit.payload["price"],
        "brand": hit.payload["brand"],
        "category": hit.payload["category"],
        "score": hit.score
    } for hit in results]

### Семантичний пошук через OpenRouter ###
async def semantic_search(query: str, products_data: list, top_k: int = 3):
    context = "\n".join(
        f"{idx}. {p['name']} ({p['brand']}): {p['description']} | ${p['price']}"
        for idx, p in enumerate(products_data, 1)
    )
    
    prompt = f"""Аналізуй список товарів та знайди {top_k} найбільш відповідних для запиту:
    
    Запит: {query}
    
    Доступні товари:
    {context}
    
    Поверни JSON з індексами {top_k} найбільш релевантних товарів у порядку відповідності.
    Формат: {{"results": [idx1, idx2, idx3]}}"""
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "anthropic/claude-3-haiku",
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"}
    }
    
    async with httpx.AsyncClient() as http_client:
        try:
            response = await http_client.post(
                OPENROUTER_API_URL,
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            
            indices = json.loads(result["choices"][0]["message"]["content"])["results"]
            return [products_data[i-1] for i in indices]  # -1 бо індексація з 1
        except Exception as e:
            print(f"Помилка семантичного пошуку: {e}")
            return []

### Комбінований пошук ###
async def hybrid_search(query: str, top_k: int = 3):
    # Векторний пошук
    vector_results = search_products(query, top_k)
    
    # Семантичний пошук
    semantic_results = await semantic_search(query, products, top_k)
    
    # Об'єднання результатів (унікальні товари)
    combined = {}
    for p in vector_results + semantic_results:
        if p["name"] not in combined:
            combined[p["name"]] = p
    
    return list(combined.values())[:top_k]

# Головний цикл
async def main():
    while True:
        query = input("\nПошук (або 'exit'): ")
        if query.lower() == 'exit':
            break
            
        print("\n1. Векторний пошук")
        print("2. Семантичний пошук")
        print("3. Гібридний пошук")
        choice = input("Оберіть тип пошуку: ")
        
        if choice == "1":
            results = search_products(query)
        elif choice == "2":
            results = await semantic_search(query, products)
        elif choice == "3":
            results = await hybrid_search(query)
        else:
            print("Невірний вибір")
            continue
            
        if not results:
            print("Нічого не знайдено")
        else:
            print("\nРезультати:")
            for idx, item in enumerate(results, 1):
                print(f"{idx}. {item['name']} ({item['brand']}) - ${item['price']}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())