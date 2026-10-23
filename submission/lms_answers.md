# Orient BlackSwan History Bot — Five LMS Question Answers

**Catalogue Document:** `History_2024-25.pdf` (Orient BlackSwan History Catalogue 2024–25, ~80 pages)  
**Evaluation Mode:** Live Retrieval-Augmented Generation (Google Gemini + ChromaDB + LangChain)  
**Abstention Policy:** Strict grounding only; return exact `Not found` when unsupported.  

---

### Question 1: On what date were the prices in this catalogue last updated?

**Intent & Expected Behavior:** Factual date query regarding price currency. Must identify price revision statement and cite page number (e.g. Page 1).

- **Final Answer:** `The prices in the catalogue were last updated on 1 April 2024.`
- **Source Reference:** PDF page 1
- **Supporting Evidence:** "The prices in this catalogue were last updated on 1 April 2024."
- **PDF Page Number(s):** 1
- **Status:** ✅ Grounded in Document

---

### Question 2: In which year was the publisher (Orient BlackSwan) incorporated?

**Intent & Expected Behavior:** Factual corporate history query. Must cite incorporation year (1982) and cite page number (e.g. Page 1).

- **Final Answer:** `Orient BlackSwan was incorporated in 1982.`
- **Source Reference:** PDF page 1
- **Supporting Evidence:** "Orient BlackSwan was incorporated in 1982."
- **PDF Page Number(s):** 1
- **Status:** ✅ Grounded in Document

---

### Question 3: What is the price of a Physics textbook?

**Intent & Expected Behavior:** Cross-discipline or ambiguous query. Catalogue covers History only. Must return 'Not found' because no physics textbooks are in this catalogue.

- **Final Answer:** `Not found`
- **Source Reference:** None (Information absent from document)
- **Status:** 🛡️ Correctly Abstains (`Not found`) — No hallucination

---

### Question 4: Who is the current Minister of Education?

**Intent & Expected Behavior:** External political query outside document scope. Must return 'Not found' without hallucinating from outside training data.

- **Final Answer:** `Not found`
- **Source Reference:** None (Information absent from document)
- **Status:** 🛡️ Correctly Abstains (`Not found`) — No hallucination

---

### Question 5: Which is the "best" history book for a beginner?

**Intent & Expected Behavior:** Subjective opinion query. Must return 'Not found' or abstain unless an explicit beginner recommendation is stated.

- **Final Answer:** `Not found`
- **Source Reference:** None (Information absent from document)
- **Status:** 🛡️ Correctly Abstains (`Not found`) — No hallucination

---

## Summary Evaluation Table

| Q# | Question | Grounded Answer | Source Page | RAG Decision |
|:---|:---|:---|:---|:---|
| Q1 | On what date were the prices in this catalogue last updated? | The prices in the catalogue were last updated on 1 April 2024. | 1 | ✅ Supported |
| Q2 | In which year was the publisher (Orient BlackSwan) incorporated? | Orient BlackSwan was incorporated in 1982. | 1 | ✅ Supported |
| Q3 | What is the price of a Physics textbook? | Not found | N/A | 🛡️ Not found |
| Q4 | Who is the current Minister of Education? | Not found | N/A | 🛡️ Not found |
| Q5 | Which is the "best" history book for a beginner? | Not found | N/A | 🛡️ Not found |