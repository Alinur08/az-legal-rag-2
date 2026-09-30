"""Interactive demo script for mentor presentations.

Provides an interactive CLI and automated benchmark demonstrating ingestion
statistics, validation checks, and hybrid retrieval over the complete
Azerbaijani Labour Code.
"""

import argparse
import sys
from src.config import settings
from src.generation.qa_prompts import answer_question
from src.guardrails import is_relevant
from src.ingestion.validate_parser import validate_articles_file
from src.retrieval.retriever import LegalRetriever


def print_banner():
    """Prints presentation header."""
    print("=" * 70)
    print("   AZ-LEGAL-RAG: ƏMƏK MƏCƏLLƏSİ RAG ASSİSTENTİ (DEMO & EVAL)")
    print("   Phases 0-5: Ingestion, Chunking, Retrieval, Guardrails & QA")
    print("=" * 70)


def run_benchmark():
    """Executes predefined realistic legal queries and displays results."""
    print_banner()
    print("\n[1] VALIDATION SPOT-CHECKS (Data Quality Verification):")
    passed, report = validate_articles_file(settings.ARTICLES_JSON)
    for line in report:
        print(f"  {line}")

    print("\n[2] LIVE HYBRID RETRIEVAL & GROUNDED QA BENCHMARK:")
    print("    (Dense bge-m3 + BM25 Fusion + Gemini LLM Grounded Generation)")
    test_queries = [
        (
            "İşçilərin ştat ixtisarı",
            "işçilərin sayı və ya ştatları ixtisar edildikdə müqaviləyə xitam"
        ),
        (
            "Hərbi xidmət təminatları",
            "işçi hərbi xidmətə çağırıldıqda hansı təminatlar verilir?"
        ),
        (
            "Əmək haqqının gecikdirilməsi",
            "əmək haqqı hansı müddətdə ödənilməlidir, gecikmə halında nə olur?"
        ),
        (
            "Məzuniyyət hüququ və müddəti",
            "əsas əmək məzuniyyətinin minimum müddəti neçə gündür?"
        ),
        (
            "Sınaq müddəti şərtləri",
            "işə qəbul zamanı sınaq müddəti ən çox nə qədər ola bilər?"
        ),
        (
            "Qeyri-müvafiq sual (Guardrail testi)",
            "tort necə bişirilir və ya sobada neçə dərəcədə bişməlidir?"
        ),
    ]

    retriever = LegalRetriever()

    for idx, (label, query) in enumerate(test_queries, start=1):
        print("\n" + "-" * 70)
        print(f"CASE {idx}: {label}")
        print(f"Sual (Query): \"{query}\"")
        results = retriever.retrieve(query, top_k=2)

        if not results or not is_relevant(query, results):
            print("\n  🛡️ [GUARDRAIL İŞƏ DÜŞDÜ: Qeyri-müvafiq sual bloklandı]")
            print(f"  {settings.NO_ARTICLE_FALLBACK_AZ}")
            continue

        for r in results:
            print(f"\n  ▶ Maddə {r.article_number}: {r.title}")
            print(
                f"    Ümumi Hibrid Bal: {r.score:.3f} | "
                f"Dense (bge-m3): {r.dense_score:.3f} | "
                f"BM25: {r.bm25_score:.3f}"
            )
            # Show snippet
            snippet_lines = r.text.split("\n")
            body_preview = " ".join(snippet_lines[1:])[:220]
            print(f"    Məzmun (Snippet): {body_preview}...")

        if settings.GEMINI_API_KEY:
            qa_res = answer_question(
                query,
                retrieve_fn=lambda q, top_k=None, _r=results: _r,
            )
            print("\n  🤖 [PHASE 5: GEMINI ƏSASLANDIRILMIŞ CAVAB]")
            print(f"  {qa_res.answer}")
        else:
            print(
                "\n  ℹ️ [PHASE 5 QA]: GEMINI_API_KEY (.env) təyin "
                "edildikdə canlı generativ cavab aktivləşir."
            )

    print("\n" + "=" * 70)
    print("DEMO TAMAMLANDI: Bütün maddələr retrieval ilə əsaslandırılmışdır.")
    print("=" * 70)


def run_interactive():
    """Interactive loop allowing the mentor to type custom questions."""
    print_banner()
    print("İnteraktiv Rejim: Sualınızı daxil edin (çıxış: 'q')\n")
    retriever = LegalRetriever()

    while True:
        try:
            query = input("\nSualınız: ").strip()
            if not query:
                continue
            if query.lower() in ("q", "quit", "exit", "çıx"):
                print("Çıxılır...")
                break

            results = retriever.retrieve(
                query, top_k=settings.RETRIEVAL_TOP_K
            )
            if not results or not is_relevant(query, results):
                print("\n  🛡️ [QORUYUCU MEXANİZM (GUARDRAIL) İŞƏ DÜŞDÜ]")
                print(
                    "  Sualın Əmək Məcəlləsi ilə əlaqəsi "
                    "və ya etibarlılıq balı kifayət deyil."
                )
                print(f"  {settings.NO_ARTICLE_FALLBACK_AZ}")
                print(f"\n{settings.LEGAL_DISCLAIMER_AZ}")
                continue

            print(f"\nTapılmış nəticələr ({len(results)} ədəd):")
            for r in results:
                print(f"\n  [Maddə {r.article_number}] {r.title}")
                print(
                    f"  Bal: {r.score:.3f} (Dense: {r.dense_score:.3f}, "
                    f"BM25: {r.bm25_score:.3f}) | Səhifə: {r.page_start}"
                )
                first_lines = r.text.split("\n")[:4]
                print("  " + "\n  ".join(first_lines))

            if settings.GEMINI_API_KEY:
                print("\n  🤖 Əsaslandırılmış hüquqi cavab hazırlanır...")
                qa_res = answer_question(
                    query,
                    retrieve_fn=lambda q, top_k=None, _r=results: _r,
                )
                print("\n" + "=" * 70)
                print("🤖 ƏSASLANDIRILMIŞ HÜQUQİ CAVAB (PHASE 5):")
                print(qa_res.answer)
                print("=" * 70)
            else:
                print(f"\n{settings.LEGAL_DISCLAIMER_AZ}")
        except (KeyboardInterrupt, EOFError):
            print("\nÇıxılır...")
            break


def main():
    """CLI entrypoint for demo presentation."""
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Demo and presentation script for az-legal-rag."
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run interactive search prompt for live mentor questions."
    )
    args = parser.parse_args()

    if args.interactive:
        run_interactive()
    else:
        run_benchmark()


if __name__ == "__main__":
    main()
