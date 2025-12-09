# Archive Branch - Full Project State

This branch preserves the complete state of the project as of December 9, 2025, including:

## Contents

- ✅ All evaluation runs (eval_new, eval_fixed, eval_fixed2, eval_touch, eval_final, eval_2m, eval_vga_video_gen)
- ✅ All training checkpoints and models
- ✅ Complete history in `old/` directory
- ✅ VGA comparison files and documentation
- ✅ All analysis markdown files

## Purpose

This branch serves as a **complete historical archive** for reference. If you need to review:
- How bugs were fixed during evaluation
- What each evaluation run tested
- Complete training progression
- VGA comparison documents

**Note:** This branch is for archival purposes only. For active development:
- Use `drl-baseline` for pure DRL work
- Use `feature/vga-integration` for VGA implementation
- Use `main` for the latest stable version

## Evaluation History

| Eval Directory | Date | Success Rate | Purpose |
|----------------|------|--------------|---------|
| eval_new | Dec 6 | 8.3% | Initial broken evaluation |
| eval_touch | Dec 6 | 8.3% | First fix attempt |
| eval_fixed | Dec 6 | 50% | Environment type fixed |
| eval_fixed2 | Dec 6 | 75% | Extended test |
| eval_final | Dec 6 | 65% | Full 20-episode test |
| eval_2m | Dec 6 | 90% | Major breakthrough |
| eval_vga_video_gen | Dec 7 | 95% | Final production run |

---

**Last Updated:** December 9, 2025  
**Commit:** Initial archive snapshot
