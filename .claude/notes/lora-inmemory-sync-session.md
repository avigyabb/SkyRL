# Design session: in-memory LoRA weight sync for Megatron

Transcript of the Claude Code conversation (2026-09-14 to 2026-09-16) that scoped issue #1336,
reviewed the existing disk-based adapter sync, and produced the `sync_mode=memory` implementation
on this branch. Tool calls and their outputs are omitted; only the user's messages and the
assistant's replies are kept, in order. Paths under `/tmp/claude-1000/...` refer to a scratch
directory on the original workspace and no longer exist.

Read this to pick up the reasoning behind the design (transport x target axes, the vLLM runtime
patch, dedupe by aliasing, GPU-resident adapter), what was validated (CPU suites only) and what
was not (any GPU run), and the open questions about the alternative designs it was compared with.

---

## User

@Avi Basnet In prep for Wed's meeting with Trajectory, can you scope out this issue: https://github.com/NovaSky-AI/SkyRL/issues/1336

We wanted to do in-memory weight sync for LoRA to avoid write to disk, which can be pretty slow for large models llike Kimi K2.6.

When I looked into this before, it needed some changes in vLLM To add more details, trajectory is also reporting slow weight sync times, and a large part of it is just disk write + read (400s with NFS for GLM 5.3 on 2 B300 nodes) (edited) -- can you give me information about this issues

## Assistant

I'll pull the GitHub issue, then trace SkyRL's current LoRA weight-sync path and the vLLM side to scope what in-memory sync needs.

## User

[Image: original 2904x1430, displayed at 2000x985. Multiply coordinates by 1.45 to map to original image.]

## User

Approach this as the design lead at a small studio known for their versatility, giving every client a visual identity pitched at the treatment the task actually calls for. Make deliberate choices about palette, typography, and layout that are specific to this subject, and avoid templated designs.

## Read the request first

Calibrate treatment, not whether to design. A doc deserves the same craft as a landing page - what changes is the treatment that craft is delivered in. Format is not part of this read: author HTML, and publish Markdown only when a loaded skill explicitly instructs it - a Markdown publish keeps its filename as its title and takes almost none of the craft below, and is never a way to save time.

Many requests call for a more utilitarian treatment: a plan, a memo, a demo. Make it polished: include real typographic hierarchy, considered spacing, and a proper palette, but avoid over-designing. Most pages do not need a flashy, gigantic hero. Keep flourishes tasteful and limited.

Some requests call for an editorial treatment: a landing page, a game, an app or tool they'll keep or share.

When unsure: a well-composed page is never the wrong answer; an over-designed visual identity sometimes is.

Fundamentals below apply to everything. The editorial process after that runs only when the read above says so.

## Fundamentals for every artifact

**Honor what's already there** Look for an existing design system first - CLAUDE.md, a tokens or theme file, existing component styles. When one exists, apply it; everything below fills gaps and never overrides. Precedence is always: the user's own words, then the project's existing system, then your choices.

**Ground it in the subject.** If the subject isn't already clear, pin it: one concrete subject, its audience, and the page's single job. The subject's own world - its materials, instruments, vernacular - is where distinctive choices come from. Whatever the treatment, carry at least one detail only this subject would have - its real units and scales, its document conventions, its terms of art - as content, not ornament; it costs a plain page nothing. Build with real content throughout, never lorem.

**Pair typefaces** Typography carries the page even when the page isn't about typography. Google Fonts is the one font host the Artifact CSP admits - link it directly (`<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=...&display=swap">`); a face from anywhere else must be inlined as a @font-face data URI or it falls back silently. Either way, declare a real fallback stack. Keep running text near 65 characters wide; set a type scale and stay on it; give headings `text-wrap: balance`, body text room to breathe, and uppercase labels a touch of letter-spacing.

**Load libraries, don't paste them.** When the page genuinely needs a library - React, a charting or highlighting package - load its UMD build from cdnjs (only the script - a library's stylesheet still has to be inlined) with one pinned `<script src="https://cdnjs.cloudflare.com/ajax/libs/...">` placed before the inline script that uses its global, instead of inlining the library's source or hand-writing a stand-in; the Artifact tool's description lists the few other script hosts the CSP admits. The page's own CSS and JS, its images and its data ship with the page. Most pages need no library at all - reach for one only when it carries real weight.

**Choose neutrals, don't default to them.** A pure mid-grey reads as unconsidered; a grey with a slight hue bias toward the page's accent reads as chosen. Pure white and near-black are fine grounds when they suit the subject - the point is that the neutral was picked, not inherited.

**Design both themes.** The page renders in the viewer's theme, and the viewer has three states, not two: an explicit choice stamps `data-theme="dark"` / `data-theme="light"` on the root element, and the default "system" setting stamps *nothing* - most viewers see the un-stamped document, where only `prefers-color-scheme` separates light from dark. Structure the CSS token-level for all three: the bare `:root` block defines the complete light palette (for a deliberately dark-first design, swap light and dark consistently through this whole pattern); `@media (prefers-color-scheme: dark)` redefines only the tokens, guarded as `:root:not([data-theme="light"])` so an explicit light choice beats a dark OS; `:root[data-theme="dark"]` redefines them again so the toggle also wins in the other direction. Style components through the tokens, never directly inside a media or `[data-theme]` block - a color whose only definition sits behind `[data-theme]` never applies in the un-stamped state, and the page renders one theme's text on the other theme's ground. Two more rules keep each theme resolving as a set: the artifact composites over a ground the viewer paints in *its* theme, so `body` must set an explicit `background` from a token - a transparent body silently borrows the host's ground; and every element that sets a color takes it from the same token set as the surface behind it, never a literal that only works in one theme. Declare every token in the bare `:root` block before any media or `[data-theme]` block redefines it - a color that exists only inside one of those blocks is the classic unreadable-artifact bug. Give the second theme the same care as the first - don't naively invert; keep contrast legible and the accent working on both grounds. A design that deliberately commits to one visual world (a neon arcade screen, a letterpress invitation) may stay single-theme - then skip the media query and stamps entirely but still paint the background and every color explicitly, so the page holds on either host ground; make it a choice, not an omission.

**Let layout do the spacing.** Lay out sibling groups with flex or grid and `gap`, not per-element margins that silently collapse or double. Keep a side gutter of at least 16px at every width - set once as side padding on `body` or one outer wrapper, whose vertical padding uses `padding-block`, never a `padding` shorthand that zeroes the sides - and let rows wrap or stack to one column at phone width (~400px). Images and any `aspect-ratio` box get `max-width: 100%`, and nothing gets a `min-width` wider than the screen; only wide tables, code and diagrams may run past it - each gets `overflow-x: auto` on its own container so the page body never scrolls sideways. Reach for `font-variant-numeric: tabular-nums` wherever digits line up in columns.

**Compose repeated things as one object.** Cards in a row, label/value pairs down a list, badges on siblings: same edges, baselines and inner padding from one to the next, and a recurring element sits in the same place on each. Let content set a container's height and pick a column count the items fill, so nothing stretches over dead space or sits alone in a row. Text that can outgrow its track wraps or scrolls in its own container; clipped text is a bug.

**Not everything is a card.** Border, fill, radius and shadow each say "separate object" - spend them by role, lifting the one thing that needs it, instead of one radius and one shadow stamped on every block, which flattens the hierarchy. Lead with big-number tiles only when those figures are the point of the page.

**Draw charts to the scale.** One scale places marks, ticks and labels, and every label names a value the chart reaches; chart text takes its color from the theme tokens so it reads in both themes; marks, labels and edges stay clear of one another and inside the drawing's bounds - in SVG, leave room in the viewBox for the outermost labels and give every drawn shape an explicit fill.

**Show the page at rest.** Everything meant to be read is visible once the page has loaded, without scrolling to trigger it - that first still frame is what a thumbnail, a shared link, and a skimming reader all get. A section may animate in, but from a visible resting state, never parked at `opacity: 0` waiting on an observer. Size a hero to what it holds, not to the viewport; a `100vh` opener pushes the page itself out of that first frame. A tool or app opens in a realistic working state - the user's real data where it exists, otherwise example rows, a loaded sample, a form someone plausibly filled, plainly marked as examples and never passed off as the user's own figures - so the first look shows what it does; an empty shell waiting for input shows nothing.

**Avoid AI-generated design** AI-generated design currently clusters around a few looks: warm cream (#F4F1EA) with a serif display and terracotta accent; near-black with a lone acid-green or vermilion pop; broadsheet hairline rules with dense columns; a purple-to-blue gradient hero on white; Inter or Space Grotesk as the "safe" face; emoji as section markers; everything centered; `rounded-lg` everywhere; accent bar/rail on rounded cards. Where the user pins down a visual direction, follow it exactly - their words always win, including when they ask for one of these looks. Where nothing is specified, don't spend that freedom on one of these defaults.

**Build cleanly** Be cognizant of overlapping elements, cascade collisions, silent font fallbacks. Close every non-void element, double-quote attributes, give keyboard focus a visible state, respect `prefers-reduced-motion`. Give every form control a stable `id` (the platform carries form values, focus and scroll across a republish). For generative or decorative graphics, reach for Canvas or WebGL rather than hand-authoring long SVG path data.

**CSS rules** When writing the CSS, watch your selector specificities. It is easy to generate classes that cancel each other out - a type-based selector like `.section` fighting an element-based one like `.cta` over padding and margins between sections. Structure the cascade so it doesn't silently undo your spacing.

**Writing the copy** Words are design material, not decoration. Write from the user's side of the screen - name things by what people recognize, not how the system is built (a person manages *notifications*, not *webhook config*). Active voice; a control says exactly what happens ("Publish", then a toast that says "Published"). Errors explain what went wrong and how to fix it - no apologies, no vagueness. Specific beats clever.

**Name the page like a product, not a caption.** The `<title>` is the artifact's name in the gallery and the browser tab, and it sets the reader's first impression of care. Give the page a real name: a short noun phrase, typically two to four words, specific to the subject - or, for a page that exists to answer one question, that question itself, which is then the page's name. Stop at the name - a title that carries its own explainer after a dash or colon reads as generated filler. The name must also identify the page among many: in the gallery it sits beside dozens of other artifacts, and a generic category label that could sit on any of them fails as a name just as surely as an appended explainer. When a candidate title pairs the name with a generic word - a greeting, a category, a page-type label - the name is the half to keep; a trim that drops the identity and keeps the generic word produces exactly the title that could sit on any page. And the rule removes explainers, it does not impose brevity: a multi-word title that already reads as one specific name is finished, and shortening it further only makes it generic. The one-sentence publish `description` is where the explanation belongs; the gallery shows it right under the title.

**Structure is information** Structural devices, numbering, eyebrows, dividers, labels, should encode something true about the content, not decorate it. Many generic designs use numbered markers (01 / 02 / 03), but that's only appropriate if the content actually is a sequence - like a real process or a typed timeline where order carries information the reader needs. Question if choices like numbered markers actually make sense before incorporating them.

**When it's a UI, not a document** A dashboard or tool is scanned and operated, not read top-to-bottom, so the craft shifts from typography to information design. Surface the summary before the detail; encode state in form as well as number - a pill, a chip, a severity stripe - so what needs attention reads at a glance. Semantic color (good / warning / critical) is separate from the accent hue and doesn't count as your accent. Give sparklines and charts the same care as type: an area fill, a faint grid, an emphasized endpoint. What's interactive should look interactive.



## Process

Start with what the viewer should be able to do on the page, not only what they will read: if it should take input, keep what people change for whoever opens it next, show live data, or ask Claude something, load the `artifact-capabilities` skill now and design around what it makes available to this user; a page that is only read needs none of that.

Before writing code, sketch a short design plan - a compact token system with color, type, and layout:
- **Color**: describe the palette as 4-6 named hex values.
- **Type**: typefaces for 2+ roles - a characterful display face used with restraint, a complementary body face, and a utility face for captions or data if needed.
- **Layout**: a layout concept in one or two sentences.

Then build, following the plan and deriving every color and type decision from it.

**Write, look once, publish.** Before publishing you may look at the rendered page once - one screenshot of the local file, or the Artifact tool's preview where it offers one - then one pass of edits for what it shows, without a second look. For a page that charts real numbers, take that look rather than skip it, and spend it on the chart. Don't build a test loop around your own file: no repeated screenshots, no pulling the script out to run it through node, no scripts that probe the DOM. That loop spends the session re-checking what a careful write already settled, while the user waits for a link. Then publish, check once any `window.claude` call the preview couldn't run, and stop: the live page is the review surface, and further polish is the user's to ask for. If the user reports something visibly broken - a clipped column, unreadable text, a control that does nothing - fix that and republish once.

**Open viewers** You don't need to do anything for viewers who already have the page open - published changes reach them automatically at their next quiet moment, with state carried where possible. If your page holds state a viewer would miss (a game, a long form), register `window.claude?.hot?.snapshot(...)` and boot through `window.claude?.hot?.ready ? window.claude.hot.ready(start) : start(window.claude?.hot?.data ?? {})`.

## When the request is editorial

The stance shifts: the client has already rejected proposals that felt templated, and is paying for a distinctive point of view. Make opinionated calls, and take one real aesthetic risk where it serves the work.

Review the design plan against the subject before building: if any part of it reads like the generic default you would produce for any similar page, revise that part, and note what you changed and why. Only after you've confirmed the plan's uniqueness do you write the code, following the revised plan exactly.

**Principles** 

- The hero is a thesis: open with the most characteristic thing in the subject's world - headline, image, live demo, interactive moment. 
- Typography carries the personality of the page. Pair the display and body faces deliberately, not the same families you would reach for on any other project, and set a clear type scale with intentional weights, widths, and spacing. Make the type treatment itself a memorable part of the design, not a neutral delivery vehicle for the content. 
- Leverage motion deliberately. Think about where and if animation can serve the subject: a page-load sequence, hover micro-interactions, ambient atmosphere. An orchestrated moment usually lands harder than scattered effects; choose what the direction calls for. However, sometimes less is more, and extra animation contributes to the feeling that the design is AI-generated. 
- Match complexity to the vision. Maximalist directions need elaborate execution; minimal directions need precision in spacing, type, and detail. Elegance is executing the chosen vision well.
- Spend your boldness in one place; keep everything around it quiet. If the accent fights the ground, shift it toward analogous or drop saturation rather than replacing it.

## Assistant

Full brief is published here: **https://claude.ai/artifact/Kh8UzUC7MBKTjm1ke3jFXp**

**The headline for Wednesday: Trajectory has already built in-memory LoRA sync.** Hersh Godse from Trajectory has three open SkyRL PRs plus companion fork PRs against vLLM and Megatron-Bridge. The issue text predates all of it, so the meeting is about landing their work, not starting from scratch.

**What the measurements say** (GLM-5.3, rank 32, trainer node and inference node each 8×B300, from PR #2195 and #2196):

| Path | Publication + first sample |
|---|---:|
| Current main, 30.77 GB safetensors | 254.7 s |
| #2195 compact file, 0.62 GB | 91.2 s |
| #2196 persistent NCCL transport | 8.9 s (cold init 724 s) |

Trajectory's 400 s figure is the same main path over NFS instead of local disk.

**Disk is not the biggest cost.** In the baseline, serialize plus write is 41 s. vLLM reading, materializing, registering and activating the adapter is 192 s. Even at 0.6 GB on disk, vLLM still spends 45 s on registration. That cost scales with the number of logical tensors, since the loader walks every per-expert LoRA key and then copies each into GPU slots per module. So a verl-style "just pass tensors" path removes the file but keeps that floor. PR #2196 gets 10× by skipping the public PEFT adapter entirely and staging each inference rank's local factors.

**Why vLLM changes are needed.** vLLM 0.28.0 has no in-memory LoRA API: the request struct is path-only and asserts a non-empty path. The internal tensor constructor exists but nothing public reaches it, and the EP expert filter lives only on the file path. verl monkeypatches the worker's LoRA manager inside the worker process. The 2024 upstream feature request was closed as stale. Trajectory's #2196 hard-requires their vLLM fork and pins the Megatron-Bridge fork in pyproject, so SkyRL cannot merge it as-is.

**Recommended stance.**
- Land #2195 quickly as the fallback. It is 72 lines, Megatron-only, and has had only bot reviews so far.
- Ask Trajectory to open the vLLM receiver-local staging PR against upstream main, and whether the Bridge rank-local export can land upstream first since it is the smaller ask.
- Decide whether to build a SkyRL-only in-memory path with no fork, which would cover FSDP and colocated runs that #2196 does not, at roughly one to two weeks.
- Raise how #2196 should fit Aaron's #2142 migration onto vLLM's trainer-send abstraction rather than adding a fifth parallel backend.

The brief has the full stage breakdown, the current SkyRL sync flow for both backends, the related Anyscale PRs (#2026, #2064, #2157, #2142, #2206), and six questions to bring to the meeting.

Sources: [SkyRL #1336](https://github.com/NovaSky-AI/SkyRL/issues/1336), [#2195](https://github.com/NovaSky-AI/SkyRL/pull/2195), [#2196](https://github.com/NovaSky-AI/SkyRL/pull/2196), [#2202](https://github.com/NovaSky-AI/SkyRL/pull/2202), [hershg/vllm#1](https://github.com/hershg/vllm/pull/1), [hershg/Megatron-Bridge#2](https://github.com/hershg/Megatron-Bridge/pull/2), [verl #4632](https://github.com/verl-project/verl/pull/4632), [NeMo-RL #1797](https://github.com/NVIDIA-NeMo/RL/pull/1797), [vLLM #4068](https://github.com/vllm-project/vllm/issues/4068), [verl #6078](https://github.com/verl-project/verl/issues/6078)

## User

We wanted to do in-memory weight sync for LoRA to avoid write to disk, which can be pretty slow for large models llike Kimi K2.6.

is there current work to do in memory weight sync right nwo?

## Assistant

Yes. There is one real in-memory implementation in flight, and it is Trajectory's, not ours.

**PR #2196 by Hersh Godse (Trajectory), opened 10 Sep, open.** It adds a `lora_nccl` backend that never touches disk. Megatron-Bridge exports each rank's local FP32 adapter factors, vLLM reports the exact shapes each TP rank consumes, SkyRL builds a static ownership plan once, and each update pushes buckets over persistent NCCL communicators into BF16 staging slots on the inference GPUs, then activates at a request boundary. Measured on GLM-5.3, rank 32, two 8×B300 nodes: 8.9 s per update versus 91 s for the compact-file path, with token-exact agreement against Megatron.

**The catches:**
- It requires two forks. The vLLM side needs `add_local_adapter` and `get_local_adapter_plan`, which exist only in hershg/vllm#1 against the pinned 0.28 base. The Megatron-Bridge side needs hershg/Megatron-Bridge#2, and the PR's pyproject points at that fork. Neither companion targets upstream main, and Hersh says upstream vLLM "has diverged substantially."
- Cold init is 724 s for the first generation while the plan and communicators are built.
- Scope is narrow: Megatron only, `merge_lora=false`, one trainer node, one inference TP group, PP1, no MTP or FP8, fixed rank and target modules. Nothing for FSDP or colocated runs.
- No maintainer review yet, only bot comments.

**Nothing on the Anyscale side is in-memory.** Everything we have shipped or opened still writes PEFT files and calls `load_lora_adapter` on a path: #2026 (merged, per-node writers and bf16 export for Kimi), #2064 (Casper's adapter-only colocated sync), #2157 (my backload skip), and Trajectory's own #2195 (disk compaction) and #2202 (HTTP upload of the files). Eric's #885 from January was the last Anyscale attempt at in-memory and it stalled on the vLLM gap. The line you have selected in `megatron_worker.py` is still the disk path.

**Upstream vLLM has nothing.** 0.28.0 has no tensor-carrying LoRA request, and the 2024 feature request was closed as stale. verl does it with a monkeypatch of the worker LoRA manager, which is the no-fork route if we want our own path for FSDP and colocated.

## User

if we just merge 2196 in does this solve the slow weight sync times that trajectory was experienceing

## Assistant

Short answer: for the exact case they reported, yes. For "just merge it," no, because it cannot be merged as-is.

**It does solve their reported case.** The 400 s NFS number is the non-colocated Megatron GLM-5.3 path on two B300 nodes. That is precisely the topology #2196 was built and measured on: trainer TP8/EP8 on one node, vLLM TP8 on the other, `merge_lora=false`. Their steady-state result is 8.9 s per update through first sample, with no file written or read, so NFS drops out entirely. Token-exact agreement and worst logprob delta of 4.6e-4 against Megatron.

**Why it is not a plain merge:**
- The runtime requires their vLLM fork. The worker calls `add_local_adapter` and `get_local_adapter_plan` and raises if they are missing. Our pin is vLLM 0.28.0 from PyPI, which has neither. Merging the SkyRL PR alone gives a backend that fails at init.
- The PR's pyproject repoints Megatron-Bridge and megatron-core at Hersh's personal fork commits. That is a dependency we would be taking on for everyone, for a backend only one topology uses.
- It is about 6,900 lines with no maintainer review yet, and it adds a fifth transport with its own control plane while Aaron's #2142 is deleting our transport layer in favour of vLLM's trainer-send abstraction.

**What it does not solve even once merged:**
- Cold init is 724 s on the first generation. For a long GLM run that amortizes. For Tinker multi-tenant use, where adapters come and go, that could be worse than the disk path unless the plan is per-layout rather than per-adapter. Worth asking.
- Nothing for colocated runs, FSDP, PP>1, multi-node trainers, MTP or FP8. Our Kimi K2.6 LoRA work runs colocated through Tinker, so #2196 does nothing for that case. The export line you have selected stays the disk path there.
- The compact-file path in #2195 remains the fallback and still has the 45 s vLLM registration floor.

**So the realistic sequence is:** merge #2195 now for a 2.8× win with no dependency changes, and treat #2196 as gated on Trajectory landing the vLLM and Bridge APIs upstream, or on us agreeing to carry those forks in the image. That is the decision to put to them on Wednesday.

## User

1. The 724 seconds is not NCCL-plan creation
The ~724–727s number is “create inference runtime through first sample.” It includes approximately:

vLLM base-weight loading: 108s
torch.compile: 169s
CUDA graph capture: 259s
KV-cache allocation and other lazy initialization
Initial LoRA publication and sampling
Only about 17.6s was the native generation-0 LoRA publication envelope.

This startup applies to both disk and native LoRA transport. It repeats when the inference runtime is destroyed and rebuilt—not whenever a new adapter is created.

Tinker already defaults to preserving the Megatron runtime:--is this true? if so can you correct the col init is.. bullet point

## Assistant

I'll check the Tinker backend to verify whether creating a new adapter reuses the existing Megatron and vLLM runtimes or rebuilds them.

Yes, that is true, and I was wrong to attribute the 724 s to the LoRA transport. The brief is corrected and republished at the same link: https://claude.ai/artifact/Kh8UzUC7MBKTjm1ke3jFXp

**What I verified in the Tinker backend** (`skyrl_train_backend.py`):
- `keep_runtime_warm_on_last_unload` defaults to `True`.
- A second or later LoRA `create_model` call registers the adapter into the already-running dispatch and returns. No Ray restart, no new vLLM engines.
- `delete_model` on a LoRA adapter unloads it from vLLM and drops the slot while the shared runtime stays up. The full teardown only happens when the flag is off on the last adapter, or on the FSDP backend, which lacks per-tenant adapter slots.
- vLLM engines are created lazily once on the first sampling call and reused after that.

So the runtime bring-up recurs only when the inference runtime is destroyed and rebuilt, exactly as Trajectory says. It is not paid per adapter, and it is identical for the disk and native paths.

**Corrected bullet**, replacing the one I gave earlier:

> The 724 s first-generation figure in #2196 is inference-runtime creation through first sample, not LoRA plan creation. Trajectory's breakdown is roughly 108 s vLLM base-weight load, 169 s torch.compile, 259 s CUDA-graph capture, plus KV allocation and lazy init. The disk path pays the same. The native LoRA transport's own generation-0 publication envelope is about 17.6 s, paid once per named adapter. Since the Tinker backend keeps the Megatron and vLLM runtime alive across adapter create and delete by default, a new adapter does not re-pay the runtime cost.

**Changes in the brief:** the TL;DR no longer cites cold init as a caveat, the stage table row is now the 17.6 s generation-0 envelope, the PR table and option C verdict drop the 724 s point, a note under the #2196 scope limits gives Trajectory's breakdown and the Tinker warm-runtime behaviour, and question 4 now asks whether the ownership plan is reusable across adapters with the same rank, targets and parallelism, which is the thing that actually matters for Tinker multi-tenant churn.

One thing I could not verify from their PR text is whether the 17.6 s is dominated by the plan build or the first NCCL rendezvous. Worth asking on Wednesday since it decides how much multi-tenant churn costs.

## User

Why it is not a plain merge:
- The runtime requires their vLLM fork. The worker calls add_local_adapter and get_local_adapter_plan and raises if they are missing. Our pin is vLLM 0.28.0 from PyPI, which has neither. Merging the SkyRL PR alone gives a backend that fails at init.
- The PR's pyproject repoints Megatron-Bridge and megatron-core at Hersh's personal fork commits. That is a dependency we would be taking on for everyone, for a backend only one topology uses.
- It is about 6,900 lines with no maintainer review yet, and it adds a fifth transport with its own control plane while Aaron's #2142 is deleting our transport layer in favour of vLLM's trainer-send abstraction.

What it does not solve even once merged:
- Cold init is 724 s on the first generation. For a long GLM run that amortizes. For Tinker multi-tenant use, where adapters come and go, that could be worse than the disk path unless the plan is per-layout rather than per-adapter. Worth asking.
- Nothing for colocated runs, FSDP, PP>1, multi-node trainers, MTP or FP8. Our Kimi K2.6 LoRA work runs colocated through Tinker, so #2196 does nothing for that case. The export line you have selected stays the disk path there.
- The compact-file path in #2195 remains the fallback and still has the 45 s vLLM registration floor. so for these reasons, if we do not care about colocated and the fallback path it solves the problem after merging right?

## Assistant

Yes, with one correction to the list you pasted and one condition on the word "merging."

**Correction first.** Drop the cold-init bullet. As covered in the last message, the 724 s is vLLM runtime bring-up that the disk path pays identically, and the Tinker backend keeps the runtime warm across adapters by default. The transport's own share is about 17.6 s once per adapter. That bullet was wrong and is no longer a reason against #2196.

**The condition.** Merging the SkyRL PR alone does not produce a working backend. The worker raises at init if `add_local_adapter` and `get_local_adapter_plan` are missing, and those exist only in Hersh's vLLM fork. So "merge #2196" really means "merge #2196 and ship Trajectory's vLLM fork and Megatron-Bridge fork in the image." If we are willing to do that, then for Trajectory's reported case the answer is yes:

- Non-colocated Megatron, trainer node and inference node, `merge_lora=false`, GLM-5.3 at TP8/EP8 to vLLM TP8 is exactly the measured configuration.
- 8.9 s per update through first sample, no file written or read, so the NFS cost disappears entirely rather than shrinking.
- Correctness was checked on that workload with token-exact agreement and worst logprob delta 4.6e-4.

**What still limits it even then**, none of which affect Trajectory's current run but matter if their setup changes:

- One trainer node only. Scaling GLM training to two trainer nodes puts them back on the disk path.
- PP1, no MTP, no FP8 on the inference side, FP32 adapter sources, fixed rank and target modules per adapter name.
- The two forks are pinned to specific commits of the pinned 0.28 base and a Bridge commit from early September. Every vLLM or Bridge bump we make has to be re-ported by someone, and Hersh has said upstream vLLM has already diverged substantially.

**The remaining objections are about cost to us, not whether it works:** 6,900 unreviewed lines, a fifth transport with its own control plane while #2142 is removing our transport layer, and carrying two forks for every user of SkyRL to serve one topology. Those are the terms to negotiate on Wednesday, with the performance question settled in their favour.

## User

is there a MUCH simpler way to get the benefits of this PR? (ok if not)

## Assistant

Yes, mostly. You can get roughly 80 to 90 percent of the benefit with a fraction of the code and no forks. The last stretch to 8.9 s is what #2196's 6,900 lines and two forks actually buy.

**What #2196 removes, in order of size** (from the 91 s compact-file run):

| Cost | Seconds | Simple fix? |
|---|---:|---|
| vLLM read, materialize, register, activate | 44.5 | Mostly, see below |
| SHA-256 hashing to find duplicate expert tensors | 25.1 | Yes, trivially |
| Bridge collective export, incl. 7.4 s CUDA to CPU wait | 12.0 | No, this is what #2196's rank-local plan avoids |
| BF16 clone on trainer | 7.0 | Yes |
| Write and read | 0.9 | Yes |

**The simpler path, in three steps:**

1. **Kill the hashing and the clone.** The duplication is structural, not accidental: with `share_expert_adapters=true` the bridge hands back the same tensor for every expert. The export loop then clones each one by name, which is what destroys the aliasing and forces #2195 to rediscover it by hashing 30 GB. Dedupe by tensor identity before cloning instead. That is a few lines in `_save_lora_adapters_and_sync` and removes about 30 s.

2. **verl-style in-memory delivery, no vLLM fork.** Ship the deduplicated adapter dict through the transports we already have, NCCL broadcast for non-colocated and CUDA IPC for colocated, into the worker extension. Inside the worker, patch the LoRA manager's load step to build the adapter from the received GPU tensors via vLLM's existing `from_lora_tensors` with `device="cuda"`, using a stub path in the request so the API server's name registry stays correct. This is what verl merged in January and it works against stock vLLM. Estimate a few hundred lines plus the GPU test, one to two weeks. Also covers FSDP and colocated, which #2196 does not.

3. **Why step 2 likely eats most of the 44.5 s.** On the file path vLLM loads every key to CPU and calls `pin_memory` on each, which allocates a fresh pinned copy per key and silently undoes the aliasing, so it pins the full logical size. Then it copies each into GPU slots. Loading straight from GPU tensors skips the CPU staging and pinning entirely and leaves only the slot copies. I have not profiled the 44.5 s split, so treat this as a strong expectation, not a measurement.

**Rough outcome:** 91 s down to somewhere around 20 to 30 s per update, with the remaining time being the bridge collective gather plus vLLM slot activation. Getting from there to 8.9 s means not gathering at all: each trainer rank sends its own shards to the inference ranks that need them, which requires vLLM to expose per-rank local shapes and Bridge to expose per-rank ownership. That is the whole content of the two fork PRs, and there is no shortcut for it.

**Two things to note:**
- The stop-gap for the NFS 400 s is even simpler. #2195 alone makes the artifact 0.6 GB, at which point NFS barely matters. Merge it now regardless.
- Step 2 is a monkeypatch of a vLLM worker class from our worker extension. It is cheap to re-port on vLLM bumps because the patched function is about 40 lines, but it is still a patch. The permanent fix is a small upstream vLLM PR adding a tensor-carrying request, which is a much easier upstream ask than the receiver-local staging API.

## User

can you switch to this branch https://github.com/NovaSky-AI/SkyRL/pull/2196

## Assistant

Fetching Trajectory's branch `optim/lora-rdt` from the `hershg` remote and checking it out.

Checked out. You are now on `optim/lora-rdt`, tracking `hershg/optim/lora-rdt`, at commit `126907a7` ("Restore typed vLLM profiler configuration"). That is the exact SkyRL head Trajectory validated in the PR.

Branch facts:

| | |
|---|---|
| Commits ahead of main | 8 |
| Files changed | 44 |
| Lines | +6,924 / -75 |

Two things to keep in mind on this branch:

- The pyproject now points Megatron-Bridge and megatron-core at Hersh's fork commits. Any `uv run` that resolves the megatron extra will pull those instead of upstream. Nothing is resolved until you run something.
- The `lora_nccl` backend will fail at init against the installed vLLM 0.28.0, since it requires the `add_local_adapter` and `get_local_adapter_plan` APIs from hershg/vllm#1. The rest of the branch, including the CPU test suites for the plan, rendezvous and transport, should run on stock vLLM.

## User

can you switch back to this branch

## Assistant

Back on `optim/lora-rdt` at `126907a7`, the PR #2196 head, tracking `hershg/optim/lora-rdt`. Working tree is clean.

For reference, the reflog shows the repo was on `pr-2068` and then `avi/fix-fb-loss-fn-output-order` in between, so another session is sharing this checkout. If you want to work on both at once without stepping on each other, a separate worktree for one of them would avoid that.

## User

can you switch back to the branch

## Assistant

Now on `avi/fix-fb-loss-fn-output-order` at `da0729ed` ("Restore input order for FSDP forward_backward loss_fn_outputs under token batching"). Working tree is clean.

If you meant the PR #2196 branch instead, it is `optim/lora-rdt` and one `git checkout` away. Since two sessions are sharing this checkout, a worktree for the #2196 review would stop the branch from flipping under either of you.

## User

yes switch to 2196

## Assistant

On `optim/lora-rdt` at `126907a7`, the PR #2196 head, tracking `hershg/optim/lora-rdt`. Tree is clean.

The `megatron_worker.py` you have open now includes the branch's 330-line addition. The `lora_nccl` path branches off inside `broadcast_to_inference_engines` when the backend is selected, ahead of the existing `_save_lora_adapters_and_sync` disk path, which is unchanged as the fallback.

## User

can you split this PR into its components and surface that ones that are easy to merge now, and also mark which ones that improve weight sync time the most. can you put this PR on my fork (not on novasky)

## Assistant

I'll start by mapping the 8 commits and 44 files into logical components, then check the fork remote setup.

Done. PR #2196 is split into seven stacked draft PRs on your fork, all authored to Hersh with your sign-off, rebased onto current main.

**The split, ranked by what to do with each:**

| Fork PR | Component | Merge now? | Speedup |
|---|---|---|---|
| [#18](https://github.com/avigyabb/SkyRL/pull/18) | Move MoE LoRA layout helper into `weight_sync.lora_layout` | Yes, pure refactor, 36 lines | None |
| [#19](https://github.com/avigyabb/SkyRL/pull/19) | Materialize `profiler_config` as vLLM `ProfilerConfig` | Yes, unrelated bug fix, 24 lines | None |
| [#20](https://github.com/avigyabb/SkyRL/pull/20) | `lora_transport` planning layer and lifecycle contracts | Yes as dead code, CPU-tested, no fork import | None alone |
| [#21](https://github.com/avigyabb/SkyRL/pull/21) | `lora_nccl` transport: plan, publication, rendezvous, staging | Yes as dead code, CPU-tested | **This is the 91 s to 8.9 s mechanism**, but inert until wired |
| [#22](https://github.com/avigyabb/SkyRL/pull/22) | Inference control plane: worker methods, endpoints, client | No, staging calls the vLLM fork API | Required to reach it |
| [#23](https://github.com/avigyabb/SkyRL/pull/23) | Megatron worker publication and config validation | No, needs Bridge fork export | Where the speedup becomes selectable |
| [#24](https://github.com/avigyabb/SkyRL/pull/24) | Repoint Megatron-Bridge and megatron-core at the forks | No | None |

The speedup is not separable from the fork dependencies. Everything through #21 could land upstream today, but nothing runs faster until #22 and #23 land, and those need hershg/vllm#1 and hershg/Megatron-Bridge#2. The honest reading for Wednesday: #18 through #21 are about 4,600 of the 6,900 lines and can be reviewed and merged on their own schedule. The remaining 2,300 lines are gated on upstream vLLM and Bridge.

**On the fork:** main is fast-forwarded to upstream `58891b2f`. Branches `pr2196/1-…` through `pr2196/7-…` are stacked, each PR's base is the previous branch, and `pr2196/full-rebased` holds Hersh's original 8 commits rebased. The rebase had one conflict, an import line in `megatron_utils.py`.

**Tests.** The 17 CPU suites #2196 touches, run against stock vLLM 0.28.0 at the stack head:

| | |
|---|---|
| Passed | 351 |
| Skipped | 2 |
| Failed | 1 |

The failure is the mxfp8 config test and fails identically on main, since its message depends on GPU architecture. No GPU tests were run.

**Two environment notes.** The shared `.venv` cannot import torch right now, with an undefined `ncclCommResume` symbol, which is independent of this branch. I ran tests in an isolated env with `--with 'ray==2.51.1'` for the cluster mismatch. The main checkout is still on `optim/lora-rdt`, and the scratch worktrees are removed.

## User

can you explain how weight syncing currently works with megatron and lora only?

## Assistant

Let me confirm two details on main before writing this up: how the merged path fuses adapters, and how the dispatch layer orders pause, offload and the adapter load.

There are two modes, chosen by `megatron_config.lora_config.merge_lora`. Everything below is what `main` does today.

**Mode 1, `merge_lora=true` (default): merged full-weight sync.** vLLM runs with LoRA disabled and serves a plain model. At sync time, Megatron-Bridge's HF export folds each adapter into its base weight, so the trainer ships full merged tensors through the normal weight-sync transport, NCCL broadcast or CUDA IPC, the same as full-parameter training. LoRA is invisible to the inference side. The cost is moving the whole base model every step, which is what makes it unworkable at GLM-5.3 or Kimi scale. It also has a small train-vs-inference numeric gap, because `W + BA` computed once is not bit-identical to `xW + x(BA)` computed separately.

**Mode 2, `merge_lora=false`: adapter-only sync via disk.** vLLM starts with `--enable-lora`, `max_lora_rank` set to the training rank, and `max_loras` slots. Only adapter tensors move. This is the path Trajectory's numbers describe, and it is entirely in `_save_lora_adapters_and_sync` in `megatron_worker.py`.

```
dispatch    save_weights_for_sampler(model_id)
            ensure_active_adapter          AdapterStore.swap_to(model_id) so the live GPU LoRA is this tenant's
            colocated: wake_up(weights) -> broadcast -> wake_up(kv_cache)
            non-colocated LoRA: no pause at all, load_lora_adapter swaps in place

worker      broadcast_to_inference_engines
            _resolve_lora_sync_target      -> (lora_name, lora_sync_path[/model_id])
            _save_lora_adapters_and_sync:
  all ranks   bridge.export_adapter_weights(actor_module)      collective TP/PP/EP gather
  writers     keep gathered tensors on CPU (bf16 clone)         others discard
  writers     fold_lora_rank_scale_for_vllm                     normalize_moe_lora rank fix
  writers     _convert_moe_experts_lora_to_vllm                 3D fused experts -> flat PEFT keys
  writers     build_adapter_config_dict                         adapter_config.json
  writers     save_file(adapter_model.safetensors) + os.replace atomic
  all ranks   barrier
  rank 0      RemoteInferenceClient.load_lora_adapter(name, path)

server      POST /skyrl/v1/load_lora_adapter
            add_lora(LoRARequest(path, load_inplace=True)), then load_inplace=False

vLLM worker LRUCacheWorkerLoRAManager._load_adapter
            PEFTHelper.from_local_dir -> LoRAModel.from_local_checkpoint (CPU, pinned)
            remove old adapter with same id -> add_adapter -> activate_adapter
            per module: set_lora copies A and B into the GPU slot
```

**Details that matter:**

- **Who writes.** With `merge_lora=false` every vLLM worker reads `lora_sync_path` from its own filesystem. Rank 0 always writes. On the first call, rank 0 drops a probe file and all ranks gather hostnames. A rank that leads its node and cannot see rank 0's probe concludes the path is node-local and writes its own copy. On a shared filesystem only rank 0 writes. That is from #2026.
- **Memory.** All ranks must join the export collectives, but only writer ranks materialize the gathered state. With per-expert MoE adapters this is tens of GB per copy, which is why non-writers pass `cpu=False` and discard.
- **The two rewrites.** vLLM's PEFT helper applies one `lora_alpha / r` to every module and has no `rank_pattern`. With `normalize_moe_lora`, experts train at rank divided by top-k, so the export multiplies those `lora_B` tensors by the rank ratio to keep the sampled policy equal to the trained one. Separately, fused expert tensors are reshaped from Megatron's 3D layout into the flat `experts.base_layer` and `experts` keys vLLM's loader accepts.
- **Multi-tenant.** Under Tinker each `model_id` gets a subdirectory under `lora_sync_path` and its own vLLM adapter name, so `sample(model=<id>)` routes correctly. `AdapterStore.swap_to` makes that tenant's adapter the live GPU state before export. On delete, the worker unloads the adapter on vLLM and removes the subdirectory.
- **Why the custom endpoint.** vLLM's stock `/v1/load_lora_adapter` with `load_inplace=true` leaves the flag set, so the engine re-read the files on every generate. SkyRL's endpoint resets it after the load. The upstream fix, vLLM PR 41482, has been open since May.
- **No pause on non-colocated.** Because `add_lora` replaces the adapter in place, the dispatch skips `pause_generation` entirely for adapter-only sync when not colocated. Colocated runs still sleep and wake the engine around it.
- **Offload interaction.** Under `colocate_all`, `_prepare_for_weight_sync` backloads the full policy model to GPU before the sync, even though adapter-only export only needs the adapters. That is what #2157 and issue #2206 are about.

**Where the time goes on GLM-5.3**, from Trajectory's profile of this exact path: about 12 s in the bridge export, 7 s cloning to CPU, 41 s writing 30 GB, and 192 s inside vLLM reading, materializing on pinned CPU memory, registering and activating. The vLLM share dominates because the loader walks every per-expert key and copies each into GPU slots.

## User

if we just do the normal weight syncing that we currently have but instead do it over memory instead of filepath, do we still get most of the improvements to weight syncing time? how much do we miss out by not implementing the local shard syncing?

## Assistant

Yes. Going in-memory with the existing public-adapter design gets most of the wall-clock win. The local-shard design in #2196 buys roughly another 10 to 20 seconds per update on top of that, and its cost is the two fork dependencies.

**Stage by stage on GLM-5.3, rank-0 seconds per update.** The first three columns are Trajectory's measurements. The fourth is my estimate for "same adapter, shipped over NCCL or CUDA IPC instead of a file", which nobody has profiled yet.

| Stage | main | #2195 compact | in-memory (est.) | #2196 |
|---|---:|---:|---:|---:|
| Bridge collective export | 11.8 | 12.0 | 5 to 12 | 0, no gather |
| Clone to CPU | 7.0 | 7.0 | 0 | 0 |
| Dedupe | 0 | 25.1 hashing | ~0 by tensor identity | n/a |
| Serialize, write, read | 40.9 + read | 0.9 + read | 0 | 0 |
| Transfer | via disk | via disk | 1 to 3 | included |
| vLLM materialize, register, activate | ~190 | 44.5 | 5 to 15 | ~2 |
| Total to first sample | 254.7 | 91.2 | **~15 to 30** | **8.9** |

**Why the in-memory estimate lands there:**

- Writing, reading and hashing disappear outright. Dedupe becomes free because the bridge hands back the same tensor object for every shared expert, and the export loop is what clones them apart today.
- The 7 s clone goes away, and most of the bridge's 7.4 s CUDA-to-CPU wait goes with it, if you keep the gathered tensors on GPU and hand them straight to the transport.
- The 44.5 s inside vLLM is mostly not compute. The file loader moves every key to CPU, pins each one separately, which re-materializes the deduplicated tensors at full size, then copies each into the GPU slots. Building the adapter from GPU-resident tensors skips the CPU round trip and pinning. What remains is a Python loop over tens of thousands of keys and one slot copy per module, which is the 5 to 15 s range. This is the number that most needs a profile before anyone commits to a plan.

**What you give up without local-shard sync:**

- **The gather.** The public adapter has to be assembled on the trainer by a collective across TP, PP and EP ranks, then sent whole to every inference rank, which then slices it back down to its own TP shard. #2196 skips both: each trainer rank sends only what it owns to the inference ranks that consume it. That is the 5 to 12 s bridge export plus the replication in the transfer.
- **vLLM's public-adapter rebuild.** Even from GPU tensors, vLLM still constructs the full `LoRAModel` and walks every module. #2196 writes each rank's factors directly into staging buffers shaped for that rank.
- **Staged activation.** #2196 stages into an inactive slot on every rank and flips at a request boundary, with rollback if any rank fails. The in-memory `add_lora` path replaces in place, the same as today's disk path.

**How much that matters.** Against the 400 s Trajectory sees on NFS, in-memory recovers roughly 93 to 96 percent of the time and #2196 recovers about 98 percent. The gap between them is 10 to 20 s per update, on a workload whose training step at 32K context is minutes. It is a real difference, but a second-order one, and the in-memory path also covers FSDP and colocated runs, which #2196 does not.

**One caveat on the in-memory path at MoE scale.** vLLM's file loader filters out non-local experts per rank under expert parallelism. `from_lora_tensors` does not, so an in-memory loader has to reimplement that filter, or every inference rank holds the whole expert adapter set transiently. At 0.6 GB deduplicated that is harmless. At the 30 GB un-deduplicated size it is not, so dedupe by identity is a requirement, not an optimization.

## User

https://github.com/NovaSky-AI/SkyRL/issues/1336 -- does this fix in vllm for lora only exist yet?

## Assistant

Let me check vLLM's current main directly, not just the pinned 0.28.0.

No. Step 2 of the issue, the vLLM fix that lets the LoRA manager load in-memory weights, does not exist upstream as of today, and nothing is open that would add it.

**vLLM `main` right now** (checked this morning, not just the 0.28.0 pin):

- `LoRARequest` still has only `lora_name`, `lora_int_id`, `lora_path`, `base_model_name`, `tensorizer_config_dict`, `load_inplace`, `is_3d_lora_weight`. No tensor field.
- `LRUCacheWorkerLoRAManager._load_adapter` still calls `PEFTHelper.from_local_dir` and `LoRAModel.from_local_checkpoint`. The path is the only input.
- `LoRAModel.from_lora_tensors` exists, but no public entry point reaches it.
- Hersh's receiver-local API from hershg/vllm#1, `add_local_adapter` and `get_local_adapter_plan`, is not in upstream `main` and has not been opened as an upstream PR.

**Nothing in flight.** A search of vLLM PRs for tensor-carrying LoRA requests, in-memory LoRA, or the fork's API names turns up only unrelated LoRA bugfixes. The one upstream request for this feature, issue #4068 from April 2024, was closed as stale in November 2024 with no implementation.

**How other RL stacks get around it:**

- **verl** carries a `TensorLoRARequest` subclass and monkeypatches `_load_adapter` at class level inside the vLLM worker process, routing to `from_lora_tensors`. Merged January 2026, still their production path.
- **verl's vllm-omni integration** did the same monkeypatch. An RFC to replace it with tmpfs plus `cudaHostRegister` was closed without action in May.
- **Trajectory** wrote the deeper receiver-local staging API in a fork, which is the right end state but is a larger upstream ask than the tensor-request patch.

**What this means for the issue.** Step 2 is still the blocker it was when Sumanth filed it in March, and step 3, upgrading to a release with the fix, cannot start. The two realistic ways forward are the ones already discussed: carry a verl-style patch from SkyRL's worker extension, which needs no vLLM change and could ship in weeks, or get Trajectory to open hershg/vllm#1 against upstream and wait for a release. Given the 2024 feature request died of inactivity, whoever wants this upstream will have to drive the PR themselves.

## User

if we just go with the normal weight syncing pattern but do it in memory is it easy to build the architecture where we specify the transport pattern (nccl, rdt, ipc, storage) and the target (full params, loras)

## Assistant

Let me look at the current abstraction boundaries on main so the answer reflects what actually exists.

Yes, and the current code is already most of the way there. The sender side is factored into exactly those two axes today. What is missing is the third axis on the receive side, which is where the LoRA disk path currently escapes the design entirely.

**What exists on `main`:**

- **Target axis, sender side: `WeightExtractor`.** `extract_weights(dtype)` yields `WeightChunk`s of named tensors. The Megatron and FSDP extractors produce full HF-layout weights. Nothing about the transport knows or cares what the names are.
- **Transport axis: `WeightTransferStrategy` and `WeightTransferSender`.** Four implementations selected by `get_transfer_strategy_cls`: NCCL broadcast, CUDA IPC, delta over storage, sharded RDT. Each takes the chunk stream and moves it.
- **The escape hatch.** LoRA with `merge_lora=false` bypasses both. The worker writes files and sends a `LoraLoadRequest`, which is a `WeightUpdateRequest` subclass carrying a path. It is neither an extractor nor a transport. That is the thing to fold back in.

**What is missing: the receive-side sink.** `NewInferenceWorkerWrap.update_weights_chunk` unconditionally calls `model.load_weights`. LoRA tensors cannot go there. They need to be accumulated, then built into a `LoRAModel` via `from_lora_tensors`, then added and activated through the worker's LoRA manager, with the API server's name registry kept in step. So the design becomes:

```
target     WeightExtractor / WeightSource      full-params | lora-adapter
transport  WeightTransferStrategy              nccl | ipc | sharded_rdt | delta(storage)
sink       receive-side applier in the worker  model.load_weights | lora manager add+activate
```

with `target` carried once in the init info so the worker picks the sink, and the same `WeightChunk` stream in between.

**How each transport fares with the LoRA target:**

| Transport | LoRA target | Why |
|---|---|---|
| NCCL broadcast | Easy | Chunks are named tensors; only the sink changes |
| CUDA IPC | Easy | Same |
| Delta over storage | Already exists, differently | Writing PEFT files and calling `load_lora_adapter` *is* the storage transport for adapters. Keep it as the fallback and register it under the same target flag rather than as a special request type |
| Sharded RDT | Hard | Its bake drives `model.load_weights` against fake tensors to build a static pull plan into model parameters. LoRA slot buffers are not model parameters, so it needs a second bake. This is close to what #2196 built |

**The concrete pieces for the easy cases:**

1. **A LoRA extractor**, roughly 150 lines. Wrap `export_adapter_weights` (Megatron) or `collect_lora_params` (FSDP), deduplicate by tensor identity before any clone, apply the rank-scale fold and the MoE key rewrite that already live in `lora_layout` and `lora_export`, keep tensors on GPU, and carry `adapter_config.json` as metadata in the first chunk.
2. **A LoRA sink in the worker extension**, roughly 200 lines. Accumulate chunks, filter non-local experts under EP the way `from_local_checkpoint` does, build via `PEFTHelper.from_dict` and `from_lora_tensors(device="cuda")`, then `add_adapter` and `activate_adapter`. Since vLLM has no public API for this, it is a verl-style override of `_load_adapter` reached through `add_lora` with a stub path, so the API server's `lora_requests` map stays correct and `model=<name>` routing keeps working.
3. **Routing**, roughly 50 lines. Derive the target from `lora.rank > 0 and not merge_lora`, put it in the init info, and select the sink. Delete `LoraLoadRequest` once the storage fallback is registered the same way.
4. **Tests**: the CPU chunk and strategy tests plus the GPU weight-sync test with a LoRA case.

**One design decision to make first.** Aaron's #2142 deletes the `WeightExtractor` and strategy layers and replaces them with vLLM 0.28's `WeightSource` and `TrainerWeightTransferEngine`. The target axis maps cleanly onto `WeightSource`, so build the LoRA extractor as a `WeightSource` if #2142 lands first. The sink hook is the same either way, because vLLM's native receive path also ends in `model.load_weights` and will need the same branch. Doing this on the layer that is about to be removed would mean doing it twice.

So: the architecture is natural, the sender side already supports it, and NCCL plus IPC with the LoRA target is about two weeks including the GPU test. RDT with the LoRA target is the hard part, and it is the part #2196 already solved in its own shape.

## User

can you try impleminting this in memory weight sync for megatron using the architecture we described on top of main and can you also make changes to vllm so that this will work. you can use hersh's PR as inspiration

## Assistant

This is a real build, so I'll work on a dedicated branch off `main` in its own worktree. First I need the exact receive-side hook points in the worker wrap and the sender flow for both transports.

## User

[Request interrupted by user]

## User

one thing im curious is if it would be easier to do a patch of vllm rather than a change within a fork, continue

## Assistant

The implementation is complete and CPU-validated. It lives on branch `avi/lora-inmemory-sync` in the worktree `/home/ray/default/skyrl-lora-inmemory-wt`, based on current `main`, uncommitted per your usual rule. Your main checkout is untouched and still on `optim/lora-rdt`.

**What it does.** Setting `trainer.policy.model.lora.sync_mode=memory` on a Megatron `merge_lora=false` run ships the adapter over the existing NCCL broadcast or CUDA IPC transport and vLLM builds the adapter from the received GPU tensors. No file is written or read. `sync_mode=disk` stays the default and is unchanged.

**The architecture, as we discussed it:**

- **Transport axis** is the existing `WeightTransferStrategy`. The two push senders now forward one extra thing, `receive_target`, into `start_weight_update`. Their private methods keep their signatures, so nothing else changed.
- **Target axis** is a new `WeightExtractor.receive_target` property. `None` means the base model. `LoraAdapterExtractor` in `weight_sync/lora_target.py` returns a dict with the adapter name, `adapter_config`, and an alias map. `MegatronLoraAdapterExtractor` plugs the bridge export, the rank-scale fold and the MoE key rewrite into it.
- **Receiver sink** in `NewInferenceWorkerWrap`: a LoRA target skips layerwise reload, clones each chunk out of the transport buffer, and at finish re-attaches aliases and stages the dict for vLLM. Rank 0 then calls the existing `/skyrl/v1/load_lora_adapter` with `in_memory: true`.
- **Dedupe replaces #2195's hashing.** Shared expert adapters are grouped structurally per EP rank and verified with `torch.equal` as the export streams, so peak memory is the unique adapter, not the 30 GB public one, and a wrong group size only costs bandwidth.

**The vLLM change is a runtime patch, not a fork.** `patches/vllm/patch_lora_in_memory.py` wraps `WorkerLoRAManager._load_adapter` and serves `skyrl-memory://<name>` requests from staged tensors through `from_lora_tensors`, with the same validation and EP filtering as the directory path. It is applied at worker import like SkyRL's other vLLM patches, against the stock 0.28.0 wheel. The same change is also written as an upstream-shaped diff, `patches/vllm/lora_in_memory_upstream.patch`, touching three vLLM files and verified to apply cleanly to 0.28.0, so it can be proposed to vLLM as-is.

**Two design decisions worth knowing:**
- The adapter is built on the GPU. vLLM only pins when the device is CPU, and same device plus dtype makes its per-key copy a no-op, so aliases keep sharing storage. The registered adapter therefore lives in GPU memory at roughly its local, undeduplicated size, instead of pinned CPU memory.
- Staged tensors stay after the load so vLLM can rebuild an LRU-evicted adapter without a resend. Unloading an adapter now also frees them on every worker.

**Validation.**

| | |
|---|---|
| New tests (extractor, dedupe, sink, patch, config) | 30 passed |
| Adjacent weight-sync, client, config suites | 406 passed, 1 pre-existing failure |

The pre-existing failure is the mxfp8 config test that fails on `main` too. Black and ruff at the repo's pinned versions are clean. `.claude/docs/weight_sync.md` documents the target axis, the flow, and the memory model.

**Not done: GPU end-to-end.** This box has one L4 and its venv cannot import torch, so I did not run Megatron plus vLLM. The first GPU run should be the weight-sync GPU test with a LoRA case, and it should log the two timings the worker now emits, export-plus-send and vLLM register, to confirm the estimate of 15 to 30 seconds on GLM-5.3. The riskiest untested piece is the per-expert module check in the patch against a real MoE model's packed module mapping.

## User

can you make this SkyRL repo reflect the state of avi/lora-inmemory-sync so I can review the changes

## Assistant

I'll move the uncommitted changes from the worktree into the main checkout on the same branch, leaving them uncommitted for review.

Done. The main checkout at `/home/ray/default/SkyRL` is now on `avi/lora-inmemory-sync` at main's head with all changes present as uncommitted working-tree edits, so `git diff` and `git status` show exactly the review surface. The temporary worktree is removed.

**What you will see:**

| | Files |
|---|---|
| Modified | 15 |
| New, untracked | 5 |
| Total | +1,681 / -25 lines |

**Suggested reading order:**

1. `skyrl/backends/skyrl_train/weight_sync/lora_target.py`, new. The target axis: receive-target dict, streaming dedupe with aliases, chunking, and the `LoraAdapterExtractor` base.
2. `weight_extractor.py`, `transfer_strategy.py`, `broadcast_strategy.py`, `cuda_ipc_strategy.py`. The protocol plumbing: one `receive_target` property forwarded into `start_weight_update`.
3. `inference_servers/new_inference_worker_wrap.py`. The receiver sink: start and finish overrides, staging of chunks for both IPC and NCCL paths.
4. `patches/vllm/patch_lora_in_memory.py`, new. The vLLM runtime patch, plus `lora_in_memory_upstream.patch`, the same change as a diff against vLLM 0.28.0.
5. `workers/megatron/megatron_worker.py`. `MegatronLoraAdapterExtractor`, the extractor construction at init, and the branch in `broadcast_to_inference_engines` ahead of the disk path.
6. `remote_inference_client.py`, `vllm_server_actor.py`. The `in_memory` flag on `load_lora_adapter` and the discard on unload.
7. `config.py`, `train/utils/utils.py`, `inference_servers/utils.py`. The `sync_mode` field and its validation.
8. Tests under `tests/backends/skyrl_train/weight_sync/test_lora_target.py` and `tests/backends/skyrl_train/inference_servers/test_lora_in_memory_receive.py`, and `.claude/docs/weight_sync.md` for the flow and memory model.

The whitespace warnings during the apply came from the upstream `.patch` file, which reproduces blank lines from vLLM's own source. Nothing is committed.

## User

can you give me an order to read these in the order of which the weight sync actually happens

## Assistant

Here is one `sync_mode=memory` LoRA sync in execution order, with a pointer for each step.

**Startup, once**

1. `skyrl/train/config/config.py:110`. The `sync_mode` field on the LoRA config; default `disk`.
2. `skyrl/train/utils/utils.py:614`. Validation: `memory` requires Megatron, `merge_lora=false`, rank above 0, and the `nccl` backend.
3. `skyrl/backends/skyrl_train/inference_servers/utils.py:241`. vLLM CLI build. The shared-filesystem warning is skipped for memory mode, since no path is used.
4. `skyrl/backends/skyrl_train/patches/vllm/patch_lora_in_memory.py:190`. Applied at worker import from `new_inference_worker_wrap.py`. Wraps vLLM's `_load_adapter` so a marker path later resolves to staged tensors.
5. `skyrl/backends/skyrl_train/workers/megatron/megatron_worker.py:1710`, then the builder at `:1716`. On `init_weight_sync_state`, the worker constructs `MegatronLoraAdapterExtractor` next to the full-weight extractor and derives `experts_per_group` from the expert count and EP size. The base class at `worker.py:503` picks the transport and creates the sender as before.

**Per sync, trainer side**

6. `skyrl/backends/skyrl_train/workers/worker_dispatch.py:704`. `save_weights_for_sampler` calls `broadcast_to_inference_engines` on every rank, unchanged.
7. `megatron_worker.py:1972`. The LoRA branch picks the in-memory publisher when the extractor exists, otherwise the disk path.
8. `megatron_worker.py:1743`. `_publish_lora_adapter_in_memory` names the adapter, calls the sender, then has rank 0 register it.
9. `skyrl/backends/skyrl_train/weight_sync/transfer_strategy.py:70`, key line at `:97`. `send` reads `receive_target` before pulling chunks.
10. `skyrl/backends/skyrl_train/weight_sync/lora_target.py:267`, then `_prepare` at `:242`. Reading the target triggers the export on every rank.
11. `megatron_worker.py:446`. `export_adapter_stream` yields the bridge's public adapter on GPU, a collective.
12. `lora_target.py:104`. `dedupe_shared_expert_adapters` consumes the stream, keeps one tensor per expert group, and records aliases.
13. `megatron_worker.py:450`. `finalize_adapter` folds the rank scale, rewrites fused expert keys, and builds `adapter_config`.
14. `lora_target.py:61`. The receive-target dict is assembled from name, config, and aliases.
15. `lora_target.py:273`, then `chunk_adapter_state` at `:151`. `extract_weights` casts to the inference dtype and cuts unique tensors into size-bounded chunks.

**Per sync, transport**

16. `broadcast_strategy.py:133` for non-colocated NCCL, or `cuda_ipc_strategy.py:152` for colocated IPC. `send_chunks` stashes the target and runs the chunk loop. NCCL uses the per-chunk path at `broadcast_strategy.py:210`.
17. `transfer_strategy.py:63`. `_start_weight_update` forwards the target to the client.
18. `remote_inference_client.py:1176`. `start_weight_update` fans out over `/collective_rpc`.
19. `new_inference_worker_wrap.py:271`. Every worker enters LoRA receive mode and skips layerwise reload.
20. `cuda_ipc_strategy.py:219` or `broadcast_strategy.py:228`. One chunk goes over the wire.
21. `new_inference_worker_wrap.py:344` for IPC or `:424` for NCCL, both landing in `_skyrl_stage_lora_weights` at `:317`. Each tensor is cloned out of the transport buffer.

**Per sync, registration**

22. `new_inference_worker_wrap.py:294`. `skyrl_finish_weight_update` expands aliases via `lora_target.py:80` and stages the dict via `patch_lora_in_memory.py:71`.
23. `megatron_worker.py:1743`, back on rank 0. `load_lora_adapter(name, in_memory=True)`.
24. `remote_inference_client.py:1276`. The client posts to every server.
25. `vllm_server_actor.py:436`, marker path at `:458`. The endpoint builds a `LoRARequest` with `skyrl-memory://<name>` and calls vLLM's `add_lora`, which travels API server to engine core to workers.
26. `patch_lora_in_memory.py:176`, then `:135`. The patched `_load_adapter` finds the staged tensors, validates them, filters non-local experts under EP, and builds the `LoRAModel` on GPU. vLLM's own `add_adapter` and `activate_adapter` then copy into the slots.

**Teardown**

27. `remote_inference_client.py:1371` and `new_inference_worker_wrap.py:313`. Unloading an adapter also discards its staged tensors on every worker.

For contrast, the disk path this replaces starts at step 7 with `_save_lora_adapters_and_sync` in the same file, which is unchanged.

## User

did you try testing this and see how much performance we retain?

## Assistant

No. I have not measured it, and I have not run the path on a GPU at all.

**What was tested:** CPU unit tests only. They cover the dedupe and alias logic, chunking, the worker-side staging and finish flow, the config validation, and the vLLM patch building a real `LoRAModel` from staged tensors with expert-parallel filtering, on CPU with a fake model manager. That checks the code paths are wired correctly. It says nothing about speed.

**What was not tested:**
- The trainer side against a real Megatron model. `export_adapter_stream`, `finalize_adapter`, and the extractor construction have never executed, since they need Megatron-Bridge and a GPU.
- The end-to-end sync through an actual vLLM engine. The `add_lora` round trip, `activate_adapter` copying into slots, and sampling against the resulting adapter have not run.
- The per-expert module check in the patch against a real MoE model's packed module mapping. This is the piece most likely to need a fix on first contact.
- Any timing.

**Why not here.** This workspace has one L4 with 23 GB, and its venv cannot import torch right now, so even a small colocated Megatron plus vLLM run is not possible without rebuilding an environment with the Megatron extra, which has its own workarounds on this box.

**So the performance numbers I gave you are estimates, not results.** The 15 to 30 second figure for GLM-5.3 came from reasoning about Trajectory's stage breakdown: the file write, hashing, and read disappear by construction, and vLLM's CPU staging and pinning are skipped because the adapter is built from GPU tensors. The parts that remain, the bridge collective export and vLLM's slot activation, are the ones I cannot bound without a run. In particular I do not know how long `from_lora_tensors` plus `activate_adapter` take over roughly 70,000 keys when the tensors are already on GPU. It could be a few seconds or it could still be tens of seconds of Python overhead.

**The honest test plan, on a machine with GPUs and a working Megatron environment:**

1. Small correctness run first. A MoE model at TP=1 with `lora.rank=8`, `merge_lora=false`, `sync_mode=memory`, non-colocated, for two or three steps. Compare sampled logprobs against the same run with `sync_mode=disk`. This flushes out the module-check and layout issues.
2. Then the GPU weight-sync test in `tests/backends/skyrl_train/gpu/gpu_ci/inference_servers/test_weight_sync.py` with a LoRA case added, on TP=1 and TP=2.
3. Then the GLM-5.3 configuration from PR #2195 on two B300 nodes, reading the two log lines the worker emits, export-plus-send and register-on-vLLM, alongside the receive-side vLLM timing. That is the only way to get a number comparable to the 91 s and 8.9 s figures.

I should have been clearer earlier that the branch is untested beyond unit tests. If you can point me at a node with GPUs and the Megatron extra installed, I can run steps 1 and 2 from there.

## User

experts_per_group here means “how many expert keys may share one identical LoRA tensor,” not “how many experts an EP rank owns.”

can you rename this to be less confusing

## Assistant

I'll rename it to `experts_per_shared_adapter`, the number of consecutive expert keys that share one adapter tensor, and reword the docstrings that described it as EP ownership.

The sed also touched an unrelated file, `skyrl/tx/models/deepseekv3.py`, which has its own `experts_per_group`. Reverting that and re-running the affected tests.

Renamed to `experts_per_shared_adapter` everywhere it belongs, and the wording now matches the meaning.

**What changed:**
- `lora_target.py`: the parameter on `dedupe_shared_expert_adapters`, `_expert_alias_group`, and `LoraAdapterExtractor.__init__`. The docstrings now define it as the number of consecutive expert keys that may carry one identical adapter tensor, with `1` disabling aliasing, and note that the EP-rank expert count is merely where that value comes from under `share_expert_adapters`.
- `megatron_worker.py:1725` area: the local variable and the constructor argument, plus the comment above it, which now says "how many consecutive expert keys carry one identical adapter tensor" and explains the `num_experts / ep_size` derivation as a consequence.
- `test_lora_target.py`: the keyword argument in every call.

**One correction along the way.** The rename initially also hit `skyrl/tx/models/deepseekv3.py`, which has an unrelated variable of the same name. I reverted that file, and it is back to its committed state with no diff.

Black and ruff are clean on the three files, and both LoRA suites pass again, 30 tests.

## User

how do other rl frameworks like miles, verl do their weight sync? is it similar to this?

## Assistant

I'll check the current source for both frameworks rather than answer from memory.

Found Miles's weight-update package. Now reading its protocol layer and LoRA handling, plus verl's mechanism, in parallel.

Yes, both are close to what I built, and Miles is the nearest match. I read the current source of both rather than relying on memory.

**Miles (SGLang, forked from slime)**

Miles factors weight sync into the same two axes. A `WeightTransferProtocol` base class owns the transport with `connect`, `begin_sync`, `send_bucket`, `finalize` hooks and a `supports_lora` flag. Four protocols implement it: CUDA IPC via SGLang's `update_weights_from_tensor` for colocated, NCCL broadcast via `update_weights_from_distributed` for disaggregated, P2P RDMA, and disk delta. The first two support LoRA, the last two do not.

The target is handled by naming, not by a separate code path. `WeightUpdater.update_weights` opens a session on the engines with `begin_weight_update(selector, sync_base)`, registers any new adapter by name and config with `register_lora_adapter`, then streams buckets through the one `send_bucket`. Adapter tensors ride the same stream under `{lora_name}:{hf_key}` names alongside base tensors. `sync_base=False` declares an adapter-only round so the engine skips its base-weight unpack and re-finalize. `end_weight_update` applies everything and can verify a per-adapter SHA-256 manifest.

The engine side is upstream in SGLang. PR #36384, merged 3 Sep 2026, adds the `/register_lora_adapter` RPC, the session scope, and a "LoRA stash": `update_weights_from_tensor` and `update_weights_from_distributed` split incoming tensors on the name prefix, base tensors load as before, prefixed tensors accumulate in a session-scoped stash, and at `end_weight_update` each stashed adapter goes through `load_lora_adapter_from_tensors` with an in-place pool-slot rewrite. Streamed adapters are never LRU-evicted because there is no file to reload from; a new name at the cap is rejected instead. Multi-LoRA publishes an immutable version per push under its own engine-side name, disaggregated only.

**verl (vLLM)**

verl is closer to how I handled the vLLM side. Colocated sync uses a bucketed transfer over per-rank ZMQ IPC sockets into `update_weights_from_ipc`; buckets of base weights load as they arrive. For LoRA, the first sync carries base plus adapter and later syncs carry the adapter only, gated by `base_sync_done`. Adapter tensors are accumulated until the last bucket, cloned out of the reused buffer, wrapped in a `TensorLoRARequest` carrying `peft_config` and `lora_tensors`, and loaded through a class-level monkeypatch of `LRUCacheWorkerLoRAManager._load_adapter` that routes to `from_lora_tensors`. The `add_lora` call happens inside the worker with a fixed id of 123 and a stub path, bypassing the API server's name registry. verl's SGLang rollout used SGLang's older `load_lora_adapter_from_tensors` route, which PR #36384 removed, so that path will need to move to the stream design.

**How the branch compares**

| | SkyRL branch | Miles | verl |
|---|---|---|---|
| Transport and target separated | Yes, `receive_target` on the extractor | Yes, `supports_lora` plus name prefix | Partly, `peft_config` flag on the IPC receiver |
| Adapter rides the base transport | Yes | Yes | Yes |
| Engine-side staging, apply at end | Yes, staged dict then `load_lora_adapter` | Yes, session stash then `end_weight_update` | Yes, accumulate until last bucket |
| Engine support | Runtime patch of vLLM | Upstream in SGLang | Runtime patch of vLLM |
| Adapter name registry kept correct | Yes, via API server | Yes, via register RPC | No, fixed id in worker |
| Shared-expert dedupe on the wire | Yes, alias map | Not that I found | No |
| Checksum verification | No | Optional manifest | No |
| Multi-adapter per session | One per sync | Several, by prefix | One |

Two things Miles does that are worth borrowing: the explicit `sync_base=False` session flag rather than inferring it from the target, and the checksum manifest, which is cheap insurance for a path that has no file to diff against. Neither Miles nor verl does anything like Trajectory's rank-local shard transfer in #2196; both send the public adapter over the base-weight transport, as this branch does.

Sources: [Miles LoRA docs](https://miles.radixark.com/docs/advanced/lora), [radixark/miles](https://github.com/radixark/miles), [Miles PR #3127](https://github.com/radixark/miles/pull/3127), [Miles v0.1 paper](https://arxiv.org/abs/2609.08368), [SGLang PR #36384](https://github.com/sgl-project/sglang/pull/36384), [SGLang PR #17159](https://github.com/sgl-project/sglang/pull/17159), [verl PR #4632](https://github.com/verl-project/verl/pull/4632), [AlphaSignal on Miles P2P sync](https://alphasignal.ai/news/radixark-s-miles-v0-1-cuts-trillion-parameter-rl-sync-time-by-86)

## User

does vllm have any checks to make sure the adapters are on disk as well?

## Assistant

No. vLLM never checks that an adapter exists on disk before the worker actually tries to read it, so the marker path is safe end to end.

**The only places a path is inspected, in vLLM 0.28.0:**

- **API server, `_check_load_lora_adapter_request`.** Checks only that `lora_name` and `lora_path` are non-empty strings, and that a name is not re-added without `load_inplace`. Nothing touches the filesystem. Our marker `skyrl-memory://<name>` is a non-empty string, so it passes. SkyRL's own endpoint bypasses this check anyway and builds the `LoRARequest` directly.
- **`LoRARequest.__post_init__`.** Asserts `lora_path` is non-empty. Same, passes.
- **Engine core and scheduler.** They forward the request object to the workers untouched and only ever key on `lora_int_id` and `lora_name`. No path handling.
- **Worker, `WorkerLoRAManager._load_adapter`.** This is the first and only place the path is resolved: `get_adapter_absolute_path`, then `PEFTHelper.from_local_dir`, then `from_local_checkpoint`. The patch intercepts at the very top of `_load_adapter`, before any of those run, so for a marker path none of them execute.

**One behavior worth knowing** if the patch were ever missing: `get_adapter_absolute_path` returns an absolute path unchanged whether or not it exists, expands `~`, and treats any other non-existent string as a Hugging Face repo id and tries to download it. So an unpatched worker given `skyrl-memory://name` would attempt a Hub download, fail, and surface `LoRAAdapterNotFoundError` through the endpoint as a clear error rather than silently doing something else. The patch is applied at worker-extension import, before model load, so this cannot happen in a SkyRL-launched engine.

**Two adjacent paths that also never touch disk for us:**
- The `LoRAResolver` plugin mechanism only runs for a request naming an adapter the API server has not registered. Ours is registered by the load call.
- LRU re-adds go back through `_load_adapter` with the same request object, which is why the patch keeps the staged tensors after the first load rather than consuming them.

Separately, while checking this I found that `vllm` is no longer installed in the workspace venv at all, on top of the earlier torch breakage. Another session appears to have re-synced it. I read the 0.28.0 source from the release tag instead. Anything GPU-side on this box will need the environment rebuilt first.

## User

can you explain the difference between this PR and what hersh did in his PR

## Assistant

The two branches solve the same problem, removing the disk round-trip for `merge_lora=false` adapter sync, but they draw the line between trainer and inference in different places. The short version: Hersh's #2196 ships each rank's shards to the ranks that consume them; this branch ships the public adapter over the existing transport and lets vLLM do what it already does with it.

**What crosses the wire**

- **#2196.** Rank-local FP32 adapter factors, straight out of Megatron-Bridge's new `export_local_adapter_weights`. No collective gather on the trainer. A static ownership plan, built once per adapter name, says which trainer rank sends which factor to which inference TP rank. Each inference rank receives only what it will hold in its own LoRA slot, already in its local shape.
- **This branch.** The public PEFT adapter, the same tensors the disk path would have written to `adapter_model.safetensors`, produced by the existing `export_adapter_weights` collective, folded and rewritten exactly as today. Deduplicated on the trainer so each shared expert adapter is sent once and re-attached by name on the receiver. Every inference worker receives the whole unique adapter and vLLM slices it per TP rank at activation, as it does for a file.

**What the inference engine has to know**

- **#2196** needs vLLM to expose per-rank local LoRA shapes and accept pre-sliced factors directly into staging buffers: `get_local_adapter_plan` and `add_local_adapter`, roughly 1,100 lines in a vLLM fork touching `model_manager.py` and the LoRA layer classes. It also needs Megatron-Bridge to expose rank-local ownership, roughly 660 lines in a Bridge fork. SkyRL's pyproject is repointed at the Bridge fork.
- **This branch** needs vLLM to build a `LoRAModel` from tensors instead of a directory. That is a 40-line wrapper on `WorkerLoRAManager._load_adapter` calling vLLM's own `from_lora_tensors`, applied as a runtime patch against the stock 0.28.0 wheel. No Bridge change. Everything downstream, validation, expert filtering, packing, slot activation, LRU, is vLLM's existing code.

**Where each design pays**

| Cost | #2196 | This branch |
|---|---|---|
| Trainer gather | None | The bridge collective, about 12 s on GLM-5.3 |
| Hashing | None | None, dedupe is structural with `torch.equal` |
| vLLM adapter rebuild | None, factors land in slots | `from_lora_tensors` plus `activate_adapter` over every key, unmeasured |
| Cold init | Plan build and communicators, about 17.6 s once per adapter | None beyond the first sync |
| Measured result | 8.9 s per update | Not measured yet, estimated 15 to 30 s |

**Scope and coverage**

- **#2196** is deliberately narrow: non-colocated only, one trainer node, one inference TP group, PP1, FP32 sources, no MTP or FP8, `normalize_moe_lora` off, fixed rank and target modules. It fails explicitly outside that. It adds a fifth transport, `lora_nccl`, with its own persistent communicators, rendezvous, request-admission gate, and commit-or-rollback control plane, about 6,900 lines.
- **This branch** works wherever the existing NCCL broadcast or CUDA IPC sender works, so colocated runs are covered, and FSDP would only need its own extractor subclass. It adds no transport. It inherits every limitation of the existing senders and of vLLM's own LoRA loading, and it adds about 1,700 lines including tests and the upstream diff.

**Lifecycle and safety**

- **#2196** stages a generation into an inactive slot on every rank and flips at a request boundary, with rollback if any rank fails, and pauses admission during the flip. It keeps the previous generation sampleable throughout.
- **This branch** replaces in place through vLLM's `load_inplace=True`, the same semantics as the disk path today. There is no cross-rank staged commit. A failure on one worker surfaces as a failed load and the previous adapter stays registered on the others.

**Where they agree**

Both keep the disk path as the fallback and default. Both leave the base-model sync untouched. Both rely on the adapter name for `model=<name>` routing through the API server. And both fold the rank scale and rewrite fused expert keys the same way, because those come from the shared export helpers.

**How to think about the choice**

#2196 is the right end state for very large MoE, because avoiding the gather and the public-adapter rebuild is where the last 10 to 20 seconds live. Its price is two forks and a large parallel control plane. This branch is the portion of the win that needs no upstream changes, covers more topologies, and slots into the transport abstraction that #2142 is about to replace, so it would carry over as a `WeightSource`. If Trajectory gets the vLLM and Bridge APIs upstream, the natural merge is to keep this branch's target axis and let `lora_nccl` become one more transport underneath it, rather than the separate backend it is today.

## User

what compute would i need to test out this new branch weight sync times to compare

## Assistant

Three tiers, depending on whether you want the plumbing verified, a meaningful relative number, or a number you can put next to Trajectory's 91 s and 8.9 s.

**Tier 1, plumbing only: 2 GPUs of any size, even L4 or A10.** Non-colocated, one trainer GPU and one vLLM GPU, a dense model like Qwen3-0.6B with rank 8. This exercises the extractor, both transports, the worker sink, the patched loader, and the `model=<name>` routing end to end. It says nothing about MoE aliasing or speed, and it would not be worth timing. This box cannot even do this today: one L4, and the venv no longer has vLLM or a working torch.

**Tier 2, real MoE and a relative number: one 8×H100 80 GB or 8×H200 node.** Qwen3-30B-A3B, or Qwen3.6-35B-A3B if you want the same family the Kimi work used, at rank 32 with `share_expert_adapters=true`. Split the node 4 trainer GPUs at TP2/EP2 and 4 vLLM GPUs at TP4, `colocate_all=false`, `weight_sync_backend=nccl`. This is the first configuration where the per-expert alias groups, the EP expert filter in the patch, and vLLM's MoE LoRA packing all run for real, which is where I expect any correctness bug to surface. Run the identical job twice, once with `sync_mode=disk` and once with `sync_mode=memory`, three updates each after a warmup, and compare the per-update sync time. The absolute seconds will be far below GLM-5.3, but the ratio between the two modes and the split between the trainer's export-plus-send line and the vLLM register line tell you whether the 15 to 30 s estimate holds in shape.

**Tier 3, apples to apples with #2195 and #2196: two 8×B300 nodes.** That is what Trajectory measured on, and there is no shortcut. GLM-5.3 at 744B is about 1.5 TB in bf16, which does not fit a single 8×H200 node for the trainer and does not fit the vLLM side in bf16 either, so H100 or H200 would force an FP8 base and change the comparison. Configuration to match theirs: trainer node at TP8/EP8, inference node at vLLM TP8, 32K context, rank 32, `merge_lora=false`, `share_expert_adapters=true`, one warmup then three measured optimizer-backed updates, timing from the publish request to the first sample. Hersh's #2181 has the B300 profiling harness and profiled Tinker client he used, which is the fastest way to get identical measurement points. Kimi K2.6 at roughly 1 T needs the same two-node B300 footprint.

**What you need besides GPUs**

- Inter-node NCCL for tier 3, so InfiniBand or equivalent between the two nodes. No shared filesystem is required for memory mode, which is itself one of the things to verify.
- The Megatron extra installed and importable. On this box that historically needed the flash-attn stub and the ldconfig fix in memory, and it takes time to rebuild.
- Stock vLLM 0.28.0. No fork or image change, unlike #2196, which needs Hersh's vLLM and Bridge forks baked into the image. That is a real practical advantage for getting the run scheduled.

**What to record on every run**

- The trainer's rank-0 log line: `LoRA sync (memory): adapter ... exported+sent in Xs, registered on vLLM in Ys`.
- Publish-to-first-sample wall time, since that is what #2195 and #2196 report.
- Peak GPU memory on a trainer rank and an inference rank, because the branch holds the registered adapter on the GPU rather than in pinned host memory, and I want to confirm the dedupe keeps that at unique-adapter size rather than the 30 GB public size.
- Sampled-token agreement and logprob delta against the disk-mode run, the same check Trajectory reported, so the speed number comes with a correctness number.

If you can get me a tier 2 node with the Megatron extra working, I can run the disk versus memory comparison and the correctness check from there, and that result would tell us whether a tier 3 run is worth requesting.

## User

can you explain the cold init difference between 2196 and this branch more simply

## Assistant

Both branches pay the same big startup cost: bringing up vLLM itself. Loading the base weights, compiling, capturing CUDA graphs, allocating the KV cache. That was the 724 s in Hersh's log, and it happens once per engine no matter how adapters are synced. Set it aside.

The difference is in what each design has to do the first time it sees a particular adapter.

**#2196 builds a plan before it can send anything.** It has to learn, for this adapter's rank and target modules, which trainer rank owns which factor and which inference TP rank needs it, then agree on that plan across both sides and open persistent NCCL communicators between them. Hersh measured that first publication at about 17.6 s. Every later update for the same adapter name reuses the plan and the communicators, which is why steady state drops to 8.9 s. Change the adapter's rank, targets, or parallel layout and the plan is rebuilt.

**This branch has nothing to build.** The first sync is the same as every other sync: export, dedupe, send over the transport that was already opened for the base-model sync at startup, register with vLLM. There is no per-adapter setup step, so the first update costs the same as the tenth.

So the trade is:

| | #2196 | This branch |
|---|---|---|
| First update for an adapter | roughly 17.6 s plus a steady-state update | one steady-state update |
| Later updates | 8.9 s | estimated 15 to 30 s |
| Break-even | after two or three updates on the same adapter | |

**Why that matters for Tinker.** Tinker workloads create and delete adapters constantly, and every new adapter name is a new plan for #2196. Whether that hurts depends on whether the plan is keyed on the adapter's layout, so that a new tenant with the same rank and targets reuses it, or on the adapter name, so that every tenant pays the setup. The PR text says "once per named adapter", which suggests the latter, and that is the question I put on the list for Wednesday. For a long single-adapter run like Trajectory's GLM training, the setup is paid once and is irrelevant.

## User

how come we don't have to generate a plan in this PR

## Assistant

Because this branch never sends anything that is shaped for a particular receiver. It sends the adapter in its public form, and the receiver figures out its own slice.

**What a plan is for.** Under Megatron, each trainer rank holds only a piece of every adapter: its TP shard of the columns or rows, and only the experts its EP rank owns. Under vLLM, each inference rank likewise needs only its own TP slice of each factor. #2196 sends from one set of pieces directly to the other set of pieces. To do that it has to know, for every tensor, which sender rank has which slice and which receiver rank wants which slice, and those layouts come from two different parallel configurations that do not line up one to one. Working that mapping out, and agreeing on it on both sides so they can post matching NCCL sends and receives, is the plan. It is a property of the adapter's shape and the two parallel layouts, so it is computed once and reused.

**Why this branch does not need one.** It puts a gather on one side and a slice on the other, and both already exist.

- **Trainer side.** `export_adapter_weights` is a collective across TP, PP and EP ranks that reassembles every adapter tensor into its full, unsharded HuggingFace form. That is the same call the disk path uses to produce `adapter_model.safetensors`. After it, every tensor is complete and has a public name, and its layout no longer depends on how the trainer was parallelized.
- **Wire.** The base-model transport already knows how to move a stream of named tensors from the trainer to every inference worker. It is the same NCCL broadcast or CUDA IPC path that syncs full weights, opened once at startup. It does not care what the tensors are.
- **Inference side.** Every worker receives the whole unique adapter. When vLLM activates it, each LoRA layer's `set_lora` slices the A and B matrices to its own TP rank, and the MoE layers keep only the experts local to their EP rank. That is exactly what vLLM does when it loads an adapter from a directory, so no new logic is needed.

The dedupe and alias map do not change this. They only avoid sending the same shared expert tensor many times under different names; the receiver still ends up with the full public adapter, just with many names pointing at one storage.

**The cost of skipping the plan.** The gather on the trainer is about 12 s on GLM-5.3, and every inference worker receives and then slices more than it keeps. #2196 avoids both by knowing the mapping ahead of time. That is the whole reason it is faster in steady state, and the whole reason it needs the plan, the vLLM fork to expose local shapes, and the Bridge fork to expose rank ownership.

## User

i want to run some tests in another node, is there a way i can bring this chat context with me

## Assistant

Yes. There are three layers to carry, and you probably want all three.

**1. The code.** The branch is uncommitted in `/home/ray/default/SkyRL`, so it does not exist anywhere else yet. Two ways to move it:

- Commit and push it to your fork, then `git fetch fork avi/lora-inmemory-sync` on the other node. I have not committed anything because you have not asked. Say the word and I will commit with sign-off and push to `avigyabb/SkyRL`.
- Or carry a patch file without committing:

```bash
cd /home/ray/default/SkyRL
git add -N . && git diff --binary > ~/lora-inmemory-sync.patch
# on the other node, from main at 58891b2f:
git checkout -b avi/lora-inmemory-sync 58891b2f && git apply ~/lora-inmemory-sync.patch
```

**2. The distilled context, which is what I would rely on.** Claude Code memory for this project lives in `~/.claude/projects/-home-ray-default-SkyRL/memory/`, 21 small files. Two of them were written during this work and hold the design, the decisions, the test recipe, and what is still unvalidated: `lora-inmemory-sync-impl.md` and `lora-inmemory-sync-scoping.md`. If you copy that directory to the same path on the other node and open Claude Code in a checkout at `/home/ray/default/SkyRL`, the next session loads the index automatically. The path matters, because the project directory name is derived from the working directory.

```bash
rsync -a ~/.claude/projects/-home-ray-default-SkyRL/memory/ other-node:~/.claude/projects/-home-ray-default-SkyRL/memory/
```

**3. The full transcript, if you want to resume this exact conversation.** This session is one file:

```
~/.claude/projects/-home-ray-default-SkyRL/7cd298c2-694a-4ea3-bae0-3d14a5dad5dd.jsonl   (4.8 MB)
```

Copy it into the same directory on the other node, again with the repo at `/home/ray/default/SkyRL`, and `claude --resume` will list it. Two caveats: it replays the whole history into context, which is long at this point, and it records paths from this machine, such as the scratchpad and the removed worktree, that will not exist there. For running tests I would start a fresh session on the memory files and only reach for the transcript if you want to re-ask something specific.

**Also useful on the other node:** the published brief at https://claude.ai/artifact/Kh8UzUC7MBKTjm1ke3jFXp for the numbers and Trajectory's PRs, and the split of #2196 on your fork as PRs #18 through #24.

One environment note for wherever you land: the test recipe that worked here was an isolated env, `uv run --isolated --extra dev --extra fsdp --with 'ray==2.51.1'`, because this workspace's venv is broken. On a clean node the plain `uv run --isolated --extra dev --extra megatron` from the docs should be enough for the GPU run.

## User

can you just commit push to my fork (don't reference any other PRs), but can you also put the chat conversation in that PR so i can just pull in other node and ask the other cc to read it
