# CLAUDE.md — Legislative Bill Passage Prediction Model

## Project Purpose

Predict the probability that a specific bill in a U.S. state legislature will be enacted into law. The initial target is Ohio (136th General Assembly, 2025-2026), but the architecture must be state-agnostic and replicable for any state or session available through LegiScan.

This is not an academic exercise. The output should be a calibrated probability with interpretable factor attribution that a policy professional or lobbyist would find credible and actionable.

## Domain Context: Ohio Legislature

- **Structure**: Bicameral — Ohio House (99 members) + Ohio Senate (33 members)
- **Current session**: 136th General Assembly (2025-2026)
- **Partisan control**: Republican supermajority in both chambers (House: 65R-34D, Senate: 24R-9D). Governor: Mike DeWine (R). This is a unified trifecta with veto-proof supermajority.
- **Session cycle**: Biennial. Bills introduced in the first year carry over to the second year. Bills not passed by session end die.
- **Committee structure**: 27 House committees, 18 Senate committees, 6 joint committees (51 total)
- **Historical passage rate**: Roughly 5-15% of introduced bills are enacted in a typical Ohio session, depending on bill type. Resolutions have higher pass rates.
- **Key procedural stages**: Introduction → Committee referral → Committee hearings → Committee vote (reported out) → Floor vote in originating chamber → Sent to other chamber → Committee in second chamber → Floor vote in second chamber → Governor's desk (sign/veto/pocket)
- **Important nuance**: Many bills are "incorporated" — their language is folded into other bills (often omnibus or budget bills). The bill itself dies but its substance passes. Our model should focus on the bill-as-introduced passing, but we should note when a bill's subject matter has companion or omnibus pathways.

## Data Source: LegiScan API

- **API base URL**: `https://api.legiscan.com/?key={API_KEY}&op={operation}`
- **Tier**: Public (free), 30,000 queries/month. Be efficient — use bulk dataset downloads for historical data, reserve per-bill queries for current session monitoring.
- **Key data available**: Bill metadata, full text (base64-encoded PDF/RTF/HTML), sponsors, cosponsors, committee referrals, full history timeline with status codes, roll call votes with individual member votes, legislator profiles, session lists
- **LegiScan status codes**: 1=Introduced, 2=Engrossed, 3=Enrolled, 4=Passed, 5=Vetoed, 6=Failed. The `progress` field uses: 1=Introduced, 2=Referred to Committee (Engrossed), 3=Passed One Chamber (Enrolled), 4=Passed Both Chambers (Passed). There are also detailed `history` entries with `action` text strings.
- **change_hash**: Every bill has a `change_hash` that changes when any data about the bill is updated. Use this for efficient syncing — compare stored hashes against `getMasterListRaw` responses to identify bills needing re-fetch.
- **Bulk datasets**: Weekly ZIP archives containing all getBill, getRollCall, and getPerson payloads for a session. Use `getDatasetList` to find them, `getDataset` to download. This is far more efficient than fetching bills individually for historical data.
- **Python client**: The `legcop` PyPI package exists but may be outdated. Prefer building our own thin client for control and reliability.

## Modeling Methodology

### Literature Foundation

This model draws on established work in legislative prediction:

1. **GovTrack Prognosis** (Tauberer, 2012-present): Logistic regression on ~15 structural features per bill. Pioneered the two-stage approach (P(exits committee) and P(enacted | exits committee)). Key finding: committee dynamics and sponsor position are the strongest predictors. Trained on the prior Congress, tested on current.

2. **Nay 2016 (Skopos Labs)**: Combined bill text embeddings (word2vec) with ~12 contextual variables using an ensemble model. Key finding: text alone has predictive power, but combining text + context always outperforms either alone. Context outperforms text at time of introduction; text catches up as bills are amended.

3. **Yano, Smith, Wilkerson 2012**: "Textual Predictors of Bill Survival in Congressional Committees" — established that lexical content carries signal beyond metadata.

4. **VPF Framework (2025)**: Multi-country prediction framework achieving up to 85% precision on individual vote prediction and 84% on bill outcomes. Used legislator seniority, party affiliation, bill content features, and historical voting loyalty as key features.

5. **State-level work (ACL 2018)**: Modeled bill passage across all 50 states using lexical content + legislature/legislator features. Found that combining text and structural signals improved accuracy by 18% over state-specific baselines.

### Modeling Approach

- **Primary**: Gradient Boosted Trees (XGBoost/LightGBM). Best-in-class for tabular mixed-type features with class imbalance. Provides native feature importance.
- **Interpretable baseline**: Logistic Regression. Useful for coefficient inspection and as a sanity check.
- **Two-stage architecture**: Separate models for committee survival and final passage. This matches the actual legislative process — the dynamics that get a bill through committee are different from those that get it enacted.
- **Probability calibration**: Raw tree model outputs are typically overconfident. Apply Platt scaling (logistic) or isotonic regression to calibrate predicted probabilities against observed frequencies.
- **Class imbalance strategy**: Use `scale_pos_weight` in XGBoost (ratio of negative to positive samples), or apply SMOTE to the training set. Evaluate with precision-recall curves, not just ROC.

### Feature Hierarchy (by expected predictive power, based on literature)

**Tier 1 — Strongest predictors:**
- Bill's current progress stage (ordinal: introduced → committee → one chamber → both chambers)
- Whether sponsor is committee chair of assigned committee
- Number of cosponsors serving on the assigned committee
- Sponsor in majority party
- Bipartisan cosponsorship indicator
- Days since last action (staleness)

**Tier 2 — Strong predictors:**
- Bill type (HB/SB vs. resolutions — very different base rates)
- Total cosponsor count
- Sponsor leadership position
- Time in session (% elapsed)
- Historical committee pass-through rate
- Whether a companion bill exists in the other chamber

**Tier 3 — Moderate predictors:**
- Sponsor seniority / historical success rate
- Bill subject/topic area
- Session year (election year effects)
- Number of amendments
- Bill text length

**Tier 4 — Future enhancements (not Phase 1):**
- Bill text NLP features (topic modeling, word embeddings)
- Media attention / news mentions
- Lobbying registrations related to the bill
- Governor's public statements on the issue
- Campaign finance connections between sponsors and interest groups

### Evaluation Standards

- **Never report accuracy alone** — with 90%+ negative class, always predicting "fail" gives 90%+ accuracy
- **Primary metric**: AUC-PR (area under precision-recall curve) — sensitive to performance on the minority (passed) class
- **Calibration**: Produce calibration plots (predicted probability vs. observed frequency). A well-calibrated model saying "30% chance" should see ~30% of those bills actually pass.
- **Temporal validation only**: Always train on earlier sessions, test on later ones. Never random cross-validation across time — that leaks future information.
- **SHAP analysis**: Use SHAP (SHapley Additive exPlanations) for per-prediction feature attribution. This gives us the "why" behind each prediction, which is as important as the number itself for a policy audience.

## Coding Standards

- Python 3.11+
- Type hints on all function signatures
- Google-style docstrings
- Logging via `logging` module (not print statements)
- Configuration via `.env` for secrets, `src/config.py` for model/feature parameters
- All random operations seeded for reproducibility
- Tests for data ingestion (mock API responses), feature computation (known inputs → expected outputs), and model training (smoke tests)
- No Jupyter notebooks in the production pipeline — notebooks are for EDA only

## Important Caveats to Surface in Output

When presenting a prediction to the user, always note:

1. **This is a statistical estimate, not a deterministic forecast.** Legislative outcomes depend on political dynamics, negotiations, external events, and gubernatorial decisions that no model captures.
2. **Base rate context is essential.** A 15% probability may sound low, but if the base rate for this bill type is 4%, that bill is nearly 4x more likely to pass than average — that's a strong signal.
3. **The model cannot predict "black swan" legislative events** — sudden political crises, scandal, federal preemption, court rulings, or a governor's change of heart.
4. **Predictions are point-in-time.** The probability should be re-computed as the bill accumulates new actions (committee hearings, amendments, floor votes).
5. **State legislatures are less studied than Congress.** Most published research targets Congress. State legislative dynamics differ — smaller bodies, less media scrutiny, more influence from gubernatorial preferences and party leadership.

## Extension Points (Future Phases)

- **Phase 2**: Add NLP features (bill text embeddings, subject classification)
- **Phase 3**: Multi-state support (parameterize by state, retrain per state)
- **Phase 4**: Live monitoring dashboard (scheduled re-scoring as bills advance)
- **Phase 5**: Individual legislator vote prediction (who will vote yes/no)
- **Phase 6**: Integration with OpenStates API for additional data enrichment
