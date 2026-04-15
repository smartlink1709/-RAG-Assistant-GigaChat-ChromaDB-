"""
Главный файл для запуска RAG-ассистента с логированием.
"""

import os
import time
import logging
from typing import Optional
from dotenv import load_dotenv

from embeddings import EmbeddingStore, get_sample_documents
from rag import RAGAssistant
from cache import ResponseCache
from db_logger import DatabaseLogger
from telegram_bot import TelegramRAGBot

from pydantic_settings import BaseSettings


# =========================
# НАСТРОЙКА ЛОГИРОВАНИЯ
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("app.log", encoding="utf-8"),
        logging.StreamHandler()
    ],
    force=True
)
logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    chroma_server_nofile: bool = False


settings = Settings()


def initialize_system():
    logger.info("Инициализация системы начата")

    load_dotenv()

    gigachat_credentials = os.getenv("GIGACHAT_CREDENTIALS")
    if not gigachat_credentials:
        logger.warning("GIGACHAT_CREDENTIALS не найден")

    logger.info("Инициализация кеша")
    cache = ResponseCache(cache_file="cache.json")

    logger.info("Инициализация векторного хранилища")
    embedding_store = EmbeddingStore(
        collection_name="rag_documents",
        persist_directory="./chroma_db",
        embedding_model="Embeddings",
        credentials=gigachat_credentials,
    )

    if embedding_store.collection.count() == 0:
        logger.info("База пуста — добавляем документы")
        sample_docs = get_sample_documents()
        embedding_store.add_documents(sample_docs)
    else:
        logger.info(f"Документов в базе: {embedding_store.collection.count()}")

    logger.info("Инициализация RAG ассистента")
    rag_assistant = RAGAssistant(
        embedding_store=embedding_store,
        credentials=gigachat_credentials,
        model="GigaChat",
        temperature=0.7,
    )

    logger.info("Система успешно инициализирована")
    return embedding_store, rag_assistant, cache


def answer_question(query: str, rag_assistant: RAGAssistant, cache: ResponseCache) -> str:
    logger.info(f"Вопрос: {query}")

    # кеш
    cached_answer = cache.get(query)
    if cached_answer:
        logger.info("Ответ взят из кеша")
        print("\n💾 Ответ из кеша:")
        print("-" * 70)
        print(cached_answer)
        print("-" * 70)
        return cached_answer

    # RAG
    try:
        start_time = time.time()

        answer, search_results = rag_assistant.generate_response(
            query=query,
            top_k=3,
            verbose=True
        )

        elapsed = round(time.time() - start_time, 2)
        logger.info(f"Ответ сгенерирован за {elapsed} сек")

        cache.set(query, answer)

        print("\n💡 ОТВЕТ:")
        print("-" * 70)
        print(answer)
        print("-" * 70)

        return answer

    except Exception as e:
        logger.error(f"Ошибка при генерации ответа: {str(e)}", exc_info=True)
        return f"Ошибка: {str(e)}"


def interactive_mode(rag_assistant: RAGAssistant, cache: ResponseCache):
    logger.info("Запущен интерактивный режим")

    print("\n" + "=" * 70)
    print("💬 ИНТЕРАКТИВНЫЙ РЕЖИМ")
    print("=" * 70)
    print("\nВведите вопрос или 'exit' для выхода\n")

    while True:
        try:
            user_input = input("\n👤 Вы: ").strip()

            if user_input.lower() in ['exit', 'quit', 'выход', 'q', '']:
                logger.info("Пользователь завершил сессию")
                print("\n👋 До свидания!")
                break

            # команды
            if user_input.lower() == 'cache':
                logger.info("Команда: cache")
                print(f"\n📊 Кеш содержит {cache.size()} записей")
                continue

            if user_input.lower() == 'clear_cache':
                logger.info("Команда: clear_cache")
                cache.clear()
                print("\n✓ Кеш очищен")
                continue

            if user_input.lower() == 'stats':
                logger.info("Команда: stats")
                print(f"\n📊 СТАТИСТИКА:")
                print(f"Документов: {rag_assistant.embedding_store.collection.count()}")
                print(f"Кеш: {cache.size()}")
                print(f"Модель: {rag_assistant.model}")
                continue

            # основной запрос
            logger.info(f"User input: {user_input}")

            start_time = time.time()
            response = answer_question(user_input, rag_assistant, cache)
            elapsed = round(time.time() - start_time, 2)

            logger.info(f"Ответ отправлен | Время: {elapsed} сек")

        except KeyboardInterrupt:
            logger.info("Прерывание пользователем (Ctrl+C)")
            print("\n👋 Прервано пользователем")
            break

        except Exception as e:
            logger.error(f"Ошибка в interactive_mode: {str(e)}", exc_info=True)
            print(f"\n❌ Ошибка: {str(e)}")


def demo_mode(rag_assistant: RAGAssistant, cache: ResponseCache):
    logger.info("Запущен demo режим")

    demo_questions = [
        "Что такое Python?",
        "Что такое RAG?",
        "Что такое векторные базы данных?",
        "Что такое Python?"
    ]

    for q in demo_questions:
        answer_question(q, rag_assistant, cache)
        input("\nEnter для продолжения...")


def main():
    try:
        embedding_store, rag_assistant, cache = initialize_system()

        print("\n1 — интерактив")
        print("2 — демо")

        mode = input("Выбор: ").strip()

        if mode == "2":
            demo_mode(rag_assistant, cache)
        else:
            interactive_mode(rag_assistant, cache)

    except Exception as e:
        logger.critical(f"Критическая ошибка: {str(e)}", exc_info=True)
        print(f"\n❌ Критическая ошибка: {str(e)}")


if __name__ == "__main__":
    main()