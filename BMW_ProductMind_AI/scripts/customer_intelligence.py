"""BMW ProductMind AI - customer-intelligence CLI (Part 2).

    python -m scripts.customer_intelligence setup                 # taxonomy, mappings, clusters, insights (no LLM)
    python -m scripts.customer_intelligence estimate --sample 20  # selection + rough cost estimate (no LLM)
    python -m scripts.customer_intelligence extract --sample 20 [--category CONTROLS]
    python -m scripts.customer_intelligence extract --all --confirm-full-run   # all G60 comments (explicit)
    python -m scripts.customer_intelligence resume JOB_ID
    python -m scripts.customer_intelligence status [JOB_ID]
    python -m scripts.customer_intelligence cluster [--llm-grouping]
    python -m scripts.customer_intelligence insights [--llm] [--min-feedback 10]
    python -m scripts.customer_intelligence demo --category CONTROLS

Commands that call the LLM: extract, resume, cluster --llm-grouping, insights --llm.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import func, select  # noqa: E402

from core.ai_models import AIApiCall, AIProcessingJob, VfcNeedMapping  # noqa: E402
from core.config import ANALYSIS_VEHICLE, get_llm_settings, get_settings  # noqa: E402
from core.database import ensure_ai_schema, get_engine, session_scope  # noqa: E402
from services import ai_feedback_analysis as ai  # noqa: E402
from services.cross_source_mapping import build_cross_source_links, comments_for_need, evidence_for_need  # noqa: E402
from services.evidence_retrieval import get_feedback_by_ids  # noqa: E402
from services.insight_generator import generate_insights, insights_for_need  # noqa: E402
from services.llm_client import AnthropicLLMClient, LLMError, LLMUsage, estimate_cost_usd  # noqa: E402
from services.need_clustering import build_clusters, clusters_for_need  # noqa: E402
from services.need_taxonomy import NEED_CODES, NEED_NAMES, TAXONOMY, sync_taxonomy  # noqa: E402

V = ANALYSIS_VEHICLE


def _engine(args):
    engine = get_engine(Path(args.db).resolve())
    ensure_ai_schema(engine)
    return engine


def _llm(settings):
    if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")
            or (Path.home() / ".config" / "anthropic").exists()):
        sys.exit("No Claude API credentials found. Set ANTHROPIC_API_KEY (or run `ant auth login`) and retry.")
    return AnthropicLLMClient(settings)


def _wrap(text: str, indent: str = "    ") -> str:
    return textwrap.fill(" ".join((text or "").split()), width=110, initial_indent=indent, subsequent_indent=indent)


# ---------------------------------------------------------------------------


def cmd_setup(args) -> None:
    with session_scope(_engine(args)) as s:
        print("taxonomy:", sync_taxonomy(s))
        print("cross-source links:", build_cross_source_links(s, V))
        print("clusters:", build_clusters(s, V))
        print("insights:", generate_insights(s, V, min_unique_feedback=args.min_feedback))


def _selection(s, args) -> tuple[list[str], str]:
    if args.ids:
        return args.ids, f"explicit IDs ({len(args.ids)})"
    if args.all:
        return ai.all_feedback_ids(s, V), f"all {V} comments"
    if args.category and args.category not in NEED_CODES:
        sys.exit(f"Unknown category {args.category}. Choose from: {', '.join(sorted(NEED_CODES))}")
    ids = ai.select_sample(s, args.sample, V, need_code=args.category)
    return ids, f"deterministic sample n={args.sample}" + (f" category={args.category}" if args.category else "")


def _rough_estimate(s, ids: list[str], model: str) -> tuple[LLMUsage, float]:
    comments = ai.load_comments(s, ids, V)
    settings = get_llm_settings()
    batches = max(1, -(-len(comments) // settings.batch_size))
    system_tokens = len(ai.build_system_prompt()) // 4
    input_tokens = sum(len(ai._user_message([c])) for c in comments) // 4 + batches * 60
    usage = LLMUsage(input_tokens=input_tokens + system_tokens, cache_read_tokens=system_tokens * (batches - 1),
                     output_tokens=len(comments) * 300)  # ~2 needs + summary per comment (thinking not included)
    return usage, estimate_cost_usd(model, usage)


def cmd_estimate(args) -> None:
    settings = get_llm_settings()
    with session_scope(_engine(args)) as s:
        ids, desc = _selection(s, args)
        cached = sum(ai.is_cached(s, c, settings.model) for c in ai.load_comments(s, ids, V))
        usage, cost = _rough_estimate(s, ids, settings.model)
        full_usage, full_cost = _rough_estimate(s, ai.all_feedback_ids(s, V), settings.model)
    print(f"Selection: {desc} -> {len(ids)} comments ({cached} already cached)")
    print(f"Model {settings.model}, effort {settings.effort}, batch size {settings.batch_size}")
    print(f"Rough estimate: ~{usage.input_tokens:,} input / ~{usage.output_tokens:,} output tokens -> ~${cost:.2f}")
    print(f"Full {V} run (4,000 comments) rough estimate: ~${full_cost:.0f} "
          "(char/4 token heuristic; adaptive thinking tokens add to output cost)")


def _progress(p: ai.JobProgress) -> None:
    print(f"  batch {p.batch_index}/{p.batches_total}: processed {p.processed}/{p.total} "
          f"(cached {p.cached}, failed {p.failed}) cost so far ${p.cost_usd:.4f}", flush=True)


def cmd_extract(args) -> None:
    settings = get_llm_settings()
    if args.batch_size:
        settings.batch_size = args.batch_size
    if args.all and not args.confirm_full_run:
        sys.exit("Full processing of all comments requires --confirm-full-run.")
    llm = _llm(settings)
    with session_scope(_engine(args)) as s:
        ids, desc = _selection(s, args)
        print(f"Extracting {len(ids)} comments ({desc}) with {settings.model}; job progress is checkpointed per batch.")
        try:
            job = ai.run_extraction(s, llm, settings, ids, vehicle_code=V, selection_description=desc,
                                    allow_full_run=args.confirm_full_run, progress=_progress)
        except ai.FullRunNotConfirmed as exc:
            sys.exit(str(exc) + " Use --all --confirm-full-run.")
        except KeyboardInterrupt:
            sys.exit("Interrupted. Resume with: python -m scripts.customer_intelligence resume <JOB_ID> (see `status`).")
        except LLMError as exc:
            sys.exit(f"Stopped: {exc.kind}: {exc}")
        _print_job(job)


def cmd_resume(args) -> None:
    settings = get_llm_settings()
    llm = _llm(settings)
    with session_scope(_engine(args)) as s:
        job = ai.resume_extraction(s, llm, settings, args.job_id, progress=_progress)
        _print_job(job)


def _print_job(job: AIProcessingJob) -> None:
    print(f"{job.job_id} [{job.job_type}] {job.status}: {job.processed_items}/{job.total_items} done, "
          f"{job.cached_items} from cache, {job.failed_items} failed | {job.api_calls} API calls, "
          f"{job.input_tokens:,} in / {job.output_tokens:,} out tokens "
          f"(cache read {job.cache_read_tokens:,}, write {job.cache_write_tokens:,}) ~${job.estimated_cost_usd:.4f}")
    if job.last_error:
        print("  last error:", job.last_error[:300])


def cmd_status(args) -> None:
    with session_scope(_engine(args)) as s:
        stmt = select(AIProcessingJob).order_by(AIProcessingJob.created_at)
        if args.job_id:
            stmt = stmt.where(AIProcessingJob.job_id == args.job_id)
        jobs = s.scalars(stmt).all()
        if not jobs:
            print("No AI jobs yet.")
        for job in jobs:
            _print_job(job)
        total = s.execute(select(func.count(), func.sum(AIApiCall.estimated_cost_usd))).one()
        print(f"All API calls: {total[0]} (~${(total[1] or 0):.4f})")


def cmd_cluster(args) -> None:
    llm = _llm(get_llm_settings()) if args.llm_grouping else None
    with session_scope(_engine(args)) as s:
        print(build_clusters(s, V, llm=llm))


def cmd_insights(args) -> None:
    llm = _llm(get_llm_settings()) if args.llm else None
    with session_scope(_engine(args)) as s:
        print(generate_insights(s, V, need_code=args.category, min_unique_feedback=args.min_feedback, llm=llm))


def cmd_demo(args) -> None:
    code = args.category
    if code not in NEED_CODES:
        sys.exit(f"Unknown category {code}. Choose from: {', '.join(c.code for c in TAXONOMY)}")
    with session_scope(_engine(args)) as s:
        if not s.scalar(select(func.count()).select_from(VfcNeedMapping)):
            sys.exit("Run `setup` first.")
        print("=" * 110)
        print(f"1. CATEGORY  {code} - {NEED_NAMES[code]}  (vehicle {V} only)")
        maps = s.scalars(select(VfcNeedMapping).where(VfcNeedMapping.need_code == code)
                         .order_by(VfcNeedMapping.vfc_level2_name)).all()
        print(f"   Mapped original VFC level-2 classes: {len(maps)} "
              f"({sum(m.mapping_status == 'review_required' for m in maps)} marked review_required)")

        ids = comments_for_need(s, code, V)
        print(f"\n2. ORIGINAL FEEDBACK  {len(ids)} distinct {V} comments; first {args.examples} shown with source reference")
        sample = get_feedback_by_ids(s, ids[: args.examples], V)
        for r in sample:
            vfc = "; ".join(c.vfc_level2_3_name or "(no VFC)" for c in r.classifications)
            print(f"  [{r.original_feedback_id}] {r.country}/{r.engine_type}  {r.source.source_file} row {r.source.source_row}")
            print(_wrap(r.customer_feedback[:400]))
            print(f"    original VFC: {vfc}")

        needs = ai.extracted_needs_for(s, ids, V)
        print(f"\n3. AI-EXTRACTED CUSTOMER NEEDS  {len(needs)} need statements from "
              f"{len({n.original_feedback_id for n in needs})} of {len(ids)} comments in this category")
        if not needs:
            print(f"   none yet - run: python -m scripts.customer_intelligence extract --sample 20 --category {code}")
        for n in needs[: args.examples * 2]:
            quote = "verbatim quote" if n.quote_verified else "quote NOT verbatim"
            print(f"  [{n.original_feedback_id}#{n.need_index}] {n.need_code} | {n.polarity} | {n.confidence} | {n.vehicle_system} | {quote}")
            print(_wrap(f"need: {n.customer_need}"))
            if n.problem:
                print(_wrap(f"problem: {n.problem}"))

        clusters = clusters_for_need(s, code, V)
        print(f"\n4/5. CLUSTERS AND CALCULATED COUNTS  (counts via SQL; unique comments vs. classification rows)")
        for c in clusters[: args.clusters]:
            flag = " [review]" if c.requires_review else ""
            print(f"  {c.cluster_id} {c.cluster_level:<14} {c.label[:55]:<55} comments={c.unique_feedback_count:>4} "
                  f"classifications={c.classification_count:>4} ai_needs={c.extracted_need_count:>3}{flag}")
        if len(clusters) > args.clusters:
            print(f"  ... {len(clusters) - args.clusters} more clusters")

        ev = evidence_for_need(s, code, V)
        print(f"\n6. EVIDENCE REFERENCES")
        print(f"  DIRECT      internal BMW feedback: {ev.direct_unique_feedback_count} comments across {len(ev.direct)} VFC classes")
        for e in ev.supporting:
            flag = " [review]" if e.requires_review else ""
            print(f"  SUPPORTING  {e.source_key[:70]:<70} {e.data_origin}{flag}")
        for e in ev.contextual:
            print(f"  CONTEXTUAL  {e.source_label[:90]}")

        insights = insights_for_need(s, code, V)
        print(f"\n7. GENERATED CUSTOMER INSIGHTS  {len(insights)} for this category; top {args.insights} shown")
        for i in insights[: args.insights]:
            print(f"  {i.insight_id} ({i.cluster_id}) {i.title}   [text: {i.text_source}, validated={i.validated}]")
            print(_wrap(f"problem: {i.customer_problem}"))
            print(_wrap(f"need: {i.customer_need}"))
            print(_wrap(f"evidence: {i.evidence_summary}"))
            print(_wrap(f"internal IDs (first 8 of {len(i.internal_evidence_ids)}): {', '.join(i.internal_evidence_ids[:8])}"))
            print(_wrap(f"public: {', '.join(i.public_evidence_ids) or '-'} | studies: {len(i.study_evidence_ids)} | "
                        f"contextual: {', '.join(i.contextual_evidence_ids) or '-'}"))
            for lim in i.limitations[:4]:
                print(_wrap(f"- {lim}", "      "))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default=str(get_settings().database_path))
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("setup")
    p.add_argument("--min-feedback", type=int, default=10)
    p.set_defaults(func=cmd_setup)

    for name, func_ in (("estimate", cmd_estimate), ("extract", cmd_extract)):
        p = sub.add_parser(name)
        g = p.add_mutually_exclusive_group(required=True)
        g.add_argument("--sample", type=int)
        g.add_argument("--all", action="store_true")
        g.add_argument("--ids", nargs="+")
        p.add_argument("--category")
        p.add_argument("--confirm-full-run", action="store_true")
        p.add_argument("--batch-size", type=int)
        p.set_defaults(func=func_)

    p = sub.add_parser("resume")
    p.add_argument("job_id")
    p.set_defaults(func=cmd_resume)
    p = sub.add_parser("status")
    p.add_argument("job_id", nargs="?")
    p.set_defaults(func=cmd_status)
    p = sub.add_parser("cluster")
    p.add_argument("--llm-grouping", action="store_true")
    p.set_defaults(func=cmd_cluster)
    p = sub.add_parser("insights")
    p.add_argument("--llm", action="store_true")
    p.add_argument("--category")
    p.add_argument("--min-feedback", type=int, default=10)
    p.set_defaults(func=cmd_insights)
    p = sub.add_parser("demo")
    p.add_argument("--category", required=True)
    p.add_argument("--examples", type=int, default=3)
    p.add_argument("--clusters", type=int, default=12)
    p.add_argument("--insights", type=int, default=2)
    p.set_defaults(func=cmd_demo)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
