# Update Presentation — Requirements

Source: `presentation/Assessment-Project-Update (3).pdf` (COMP8240 Applications of Data Science, 2026 Session 2).
Read alongside the unit's "Overall Goal and Assessment Criteria" page (summarised in `instructions .md` at the repo root).

> Note: the PDF's header says "Assessment Task #1: Project Proposal". That is a template leftover. The body is the **Project Update** brief (40%), and this file treats it that way.

---

## 1. Submission logistics (hard requirements)

| Item | Requirement |
|---|---|
| **Due** | **11:55 pm, 09/10/2026 (Week 9)** |
| **Weighting** | Marked out of **40**, worth **40%** of the unit grade |
| **Format** | **Video presentation, MP4** |
| **Length** | **5 minutes**, marked on a **±10% window (4:30 to 5:30)** |
| **Face on camera** | **Required**: your face in a corner window during screen sharing |
| **Submit via** | iLearn dropbox |
| **Individual / group** | Individual |
| **Late penalty** | Standard late penalty applies |
| **Short extensions** | Short automated extensions are accepted |
| **Feedback** | Via iLearn, target ~2 weeks after submission |
| **Expected effort** | ~20 hours |
| **Use of AI** | Open. You are responsible for all AI output, and must read any AI-suggested references to check they are relevant and correct |

## 2. Purpose and learning outcomes

The semester project is about **replicating a piece of research**: get the original code running on the original data, then build **new data** and test whether the paper's claims still hold. This video is a **progress update** on the project chosen at the proposal stage.

Unit learning outcomes assessed:
- **ULO01**: Define and manage a project involving empirical research.
- **ULO02**: Apply programming and/or appropriate applications (data gathering, curation, cleaning, analysis) to practical empirical research.
- **ULO03**: Communicate a coherent argument clearly, in written and oral form, to a variety of audiences.

The task demonstrates:
1. Oral communication skills.
2. Recognising reproducible research, and the core aspects of a paper involved in reproducibility.
3. Defining and managing an empirical research project.
4. Using the tools typical of empirical computational research and how it is presented.

## 3. Which track applies to us

**Our project is a NOVEL project** (self-chosen paper: Xu et al., *How Powerful are Graph Neural Networks?*, ICLR 2019). Consequences:
- **No 80% cap.** The default project is capped at 80%; the novel project is not.
- The novel-project rubric weights apply: **6 / 8 / 8 / 8 / 10 = 40 marks** (see §6).
- We must include a brief **justification reminder**: venue, authors, code source, why we chose it.

## 4. Required content (novel project)

The video splits into two halves.

### 4.1 Recap of project (about half, perhaps slightly more)
- **Justification for project basis (reminder only, not a full re-justification):** where the work comes from: publication venue, authors, source of code, why we chose it.
- **Original dataset(s):** where they come from and why the paper uses them.
- **Replication of original work, system side:** introduce the **high-level architecture** of the system, **run the code**, and **explain the output** of the system.
- Mention **source-code availability and feasibility**.

### 4.2 Update / progress (the other half, perhaps slightly less)
- **How replication on the original data went:** Does the code work? In what environment? Did we get similar results to the paper? Say clearly what has **and has not** worked.
- **New data, description:** at least **one new existing dataset** *and* **one entirely newly constructed dataset**. Cover the source, properties, complexity and challenges. Datasets that take more work to create earn more marks.
- **New data, progress on construction:** evidence of progress in **collection, preprocessing, annotation**.
- **Wrap-up:** re-emphasise the key progress points and **what comes next**.

## 5. Suggested timing (novel project, 5:00 total)

| Time | Section | Content |
|---|---|---|
| 0:00 to 0:30 | Recap & justification | Project basis: publication venue, authors, why we chose it |
| 0:30 to 1:15 | Recap | Datasets from the original paper: where they come from, why they're used |
| 1:15 to 2:45 | Recap | High-level system architecture; source-code availability and feasibility; **run the original code and show the output** |
| 2:45 to 3:30 | Update | Replication status on original data: does the code work, in what environment, similar results? |
| 3:30 to 4:30 | Update | New datasets: **at least one existing + one being constructed**, with properties and challenges |
| 4:30 to 4:50 | Update | Progress on construction: collection, preprocessing, annotation |
| 4:50 to 5:00 | Wrap-up (10 to 15 s) | Key progress points and next steps |

## 6. Marking rubric (novel project, 40 marks)

| Component | Marks | HD (85 to 100%) looks like |
|---|---|---|
| **Recap / Justification** | **6** | Clear, concise recap **and** strong justification (source, feasibility) |
| **Replication of Original Work** | **8** | Explanation of replication progress is very clear and leaves no questions |
| **New Data: Description** | **8** | Detailed description of the new datasets (**source, properties, complexity**) that leaves no questions |
| **New Data: Progress** | **8** | Evidence of progress on the dataset being constructed, covering **collection, preprocessing, annotation**, leaving no questions |
| **Presentation Quality** | **10** | Clarity, structure, **timing balance (5 min ±10%)**, engagement and audio/visual quality are all excellent |

How the grade bands step down:
- **D (75 to 84%):** as HD, but one element weaker, or "leaves one or two questions".
- **Credit (65 to 74%):** two elements weaker, or "missing some details". For presentation, lacking in two or more areas.
- **Pass (50 to 64%):** gives "an overall picture with some detail".
- **Fail (<50%):** unclear or missing. Presentation is broadly unsatisfactory.

(Default-project weights, for reference only: 4 / 6 / 6 / 6 / 10.)

## 7. Quality criteria (from the brief)

A high-quality submission:
- [ ] is **well-structured**, with clear sections and logical flow that follow the suggested structure above;
- [ ] gives **clear descriptions of the code and the data** we are working with;
- [ ] has **enough detail to assess our progress**, including **what has and has not worked** in the replication so far;
- [ ] is **clear on both the existing datasets and the proposed new datasets**.

Delivery tips from the brief:
- [ ] Keep explanations **high-level** and avoid technical overload. Focus on **progress**.
- [ ] **Show your face in a corner window** during screen sharing (required).

## 8. Project-wide criteria this video feeds into

From the "Overall Goal and Assessment Criteria" (`instructions .md`):
- **Justification for project basis:** CORE ranking, Google Scholar venue ranking, citations.
- **Explicit inclusion of unit techniques and tools:** e.g. scripts to gather data from the Internet, data analysis, generating new data.
- **Success in replication and investigation:** get the original code running on the original data and get results close to the original. Dig into **where the method works and where it doesn't**. If we replicate only a subset, **justify** it.
- **Quality of new data and code:** readable, runnable by the marker, new data aiming for similar quality to the original.
- **GitHub repo** is part of the overall submission: report, new data, data-creation scripts.

## 9. Compliance checklist (tick before submitting)

- [ ] MP4 file, runtime between **4:30 and 5:30**
- [ ] Face visible in a corner window throughout the screen share
- [ ] Recap ≈ 2.5 to 3 min; update ≈ 2 to 2.5 min; wrap-up 10 to 15 s
- [ ] Venue (ICLR 2019, CORE A*), authors, and code source (`weihua916/powerful-gnns`) named
- [ ] Original datasets explained (IMDB-BINARY / IMDB-MULTI: origin and why they're used)
- [ ] High-level GIN architecture shown
- [ ] **Original code actually run on screen**, with its output explained
- [ ] Replication status: environment (local CPU → Colab T4 → uni RTX 6000 Ada), results vs paper, what did and didn't work
- [ ] **At least one *existing* new dataset** described (e.g. ENZYMES / DD / OGB / AIDS)
- [ ] **At least one *newly constructed* dataset** described (TMDb IMDB-style): source, properties, complexity, challenges
- [ ] Construction progress: collection, preprocessing, annotation (label source)
- [ ] Next steps stated
- [ ] TMDb attribution shown: "This product uses the TMDb API but is not endorsed or certified by TMDb."
- [ ] Submitted to the iLearn dropbox before 11:55 pm, 9 Oct 2026
