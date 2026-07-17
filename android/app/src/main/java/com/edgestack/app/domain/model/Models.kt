package com.edgestack.app.domain.model

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import java.time.LocalDate

/** One daily OHLC bar; [adj] is the dividend/split-adjusted close. */
data class SpyBar(
    val date: LocalDate,
    val open: Double,
    val high: Double,
    val low: Double,
    val close: Double,
    val adj: Double,
)

@Serializable
data class CalendarBundle(
    @SerialName("schema_version") val schemaVersion: Int = 1,
    val exchange: String = "XNYS",
    val sessions: List<String> = emptyList(),
)

/** A user-entered real position tracked by the app. */
@Serializable
data class TrackedPosition(
    val symbol: String,
    val entryPrice: Double,
    val quantity: Double,
    val entryDate: String,           // ISO yyyy-MM-dd
    val stop: Double? = null,
    val target: Double? = null,
    val horizonSessions: Int = 10,
)

@Serializable
data class TrackedPositions(val positions: List<TrackedPosition> = emptyList())

// --- paper account (GET /paper on the PC) ---
@Serializable
data class PaperPositionDto(
    val symbol: String,
    val quantity: Double = 0.0,
    @SerialName("average_fill_price") val averageFillPrice: Double = 0.0,
    @SerialName("last_price") val lastPrice: Double = 0.0,
)

@Serializable
data class PaperFillDto(
    val symbol: String,
    val session: String,
    val quantity: Double = 0.0,
    val price: Double = 0.0,
    @SerialName("transaction_cost") val transactionCost: Double = 0.0,
)

@Serializable
data class PaperReturnDto(
    val session: String,
    @SerialName("ending_equity") val endingEquity: Double = 0.0,
    @SerialName("actual_fill_return") val actualFillReturn: Double = 0.0,
    @SerialName("transaction_costs") val transactionCosts: Double = 0.0,
    @SerialName("dividend_cash") val dividendCash: Double = 0.0,
    @SerialName("financing_cash_flow") val financingCashFlow: Double = 0.0,
)

@Serializable
data class PaperStateDto(
    val cash: Double = 0.0,
    @SerialName("current_equity") val currentEquity: Double = 0.0,
    @SerialName("last_session") val lastSession: String? = null,
    val positions: List<PaperPositionDto> = emptyList(),
    val fills: List<PaperFillDto> = emptyList(),
    @SerialName("realized_returns") val realizedReturns: List<PaperReturnDto> = emptyList(),
)

@Serializable
data class EquityPoint(
    val date: String,
    val equity: Double,
    @SerialName("actual_fill_return") val actualFillReturn: Double = 0.0,
)

@Serializable
data class PaperResponse(
    val state: PaperStateDto = PaperStateDto(),
    @SerialName("equity_history") val equityHistory: List<EquityPoint> = emptyList(),
)

// --- authoritative Recommendation Engine V2 contracts ---

@Serializable
data class WeightV2(
    val symbol: String,
    val weight: Double,
    @SerialName("asset_kind") val assetKind: String,
    val sector: String = "unknown",
)

@Serializable
data class FreshnessV2(
    @SerialName("as_of") val asOf: String,
    @SerialName("expected_session") val expectedSession: String,
    @SerialName("is_fresh") val isFresh: Boolean,
    @SerialName("age_business_days") val ageBusinessDays: Int,
    val complete: Boolean,
    val compatible: Boolean,
    val reasons: List<String> = emptyList(),
)

@Serializable
data class WatchlistEntryV2(
    val symbol: String,
    @SerialName("asset_kind") val assetKind: String,
    @SerialName("horizon_sessions") val horizonSessions: Int,
    val family: String,
    val thesis: String,
    @SerialName("evidence_grade") val evidenceGrade: String,
    @SerialName("prospective_sessions") val prospectiveSessions: Int = 0,
    @SerialName("effective_resolved_outcomes") val effectiveResolvedOutcomes: Double = 0.0,
    @SerialName("zero_weight_reason") val zeroWeightReason: String,
)

@Serializable
data class SleeveContributionV2(
    @SerialName("sleeve_id") val sleeveId: String,
    @SerialName("horizon_sessions") val horizonSessions: Int,
    val family: String,
    @SerialName("expected_net_return") val expectedNetReturn: Double,
    @SerialName("effective_sample_size") val effectiveSampleSize: Double,
    @SerialName("evidence_grade") val evidenceGrade: String,
)

@Serializable
data class BaseRecommendationV2(
    val status: String,
    @SerialName("baseline_weights") val baselineWeights: List<WeightV2>,
    @SerialName("unlevered_base_weights") val unleveredBaseWeights: List<WeightV2>,
    @SerialName("promoted_sleeves") val promotedSleeves: List<SleeveContributionV2> = emptyList(),
    @SerialName("promoted_compound_sleeves")
    val promotedCompoundSleeves: List<SleeveContributionV2> = emptyList(),
    val watchlist: List<WatchlistEntryV2> = emptyList(),
    @SerialName("expected_net_return") val expectedNetReturn: Double = 0.0,
    @SerialName("expected_volatility") val expectedVolatility: Double = 0.0,
    @SerialName("turnover_estimate") val turnoverEstimate: Double = 0.0,
    val freshness: FreshnessV2,
    val warnings: List<String> = emptyList(),
)

@Serializable
data class RiskProfileV2(
    @SerialName("account_equity") val accountEquity: Double = 100_000.0,
    @SerialName("target_volatility") val targetVolatility: Double = 0.12,
    @SerialName("maximum_drawdown") val maximumDrawdown: Double = 0.15,
    @SerialName("maximum_gross_leverage") val maximumGrossLeverage: Double = 1.0,
    @SerialName("funding_spread_bps") val fundingSpreadBps: Double = 200.0,
    @SerialName("per_stock_cap") val perStockCap: Double = 0.03,
    @SerialName("sector_cap") val sectorCap: Double = 0.20,
)

@Serializable
data class RiskStateV2(
    @SerialName("state_version") val stateVersion: Int = 0,
    @SerialName("previous_effective_leverage") val previousEffectiveLeverage: Double = 0.0,
    @SerialName("peak_equity") val peakEquity: Double,
    @SerialName("current_equity") val currentEquity: Double,
    @SerialName("current_drawdown") val currentDrawdown: Double = 0.0,
    @SerialName("drawdown_state") val drawdownState: String = "NORMAL",
    @SerialName("cash_latched") val cashLatched: Boolean = false,
    @SerialName("reset_eligible") val resetEligible: Boolean = false,
    @SerialName("sessions_since_latch") val sessionsSinceLatch: Int = 0,
    @SerialName("last_session") val lastSession: String? = null,
    @SerialName("previous_target_weights") val previousTargetWeights: List<WeightV2> = emptyList(),
)

@Serializable
data class ConstraintResultV2(
    val name: String,
    @SerialName("leverage_limit") val leverageLimit: Double,
    val binding: Boolean = false,
    val detail: String = "",
)

@Serializable
data class PortfolioRecommendationV2(
    val status: String,
    @SerialName("output_risk_state") val outputRiskState: RiskStateV2,
    @SerialName("base_recommendation_weights") val baseRecommendationWeights: List<WeightV2>,
    @SerialName("personalized_target_weights") val personalizedTargetWeights: List<WeightV2>,
    @SerialName("effective_leverage") val effectiveLeverage: Double,
    val constraints: List<ConstraintResultV2> = emptyList(),
    @SerialName("binding_constraints") val bindingConstraints: List<String> = emptyList(),
    @SerialName("expected_net_return") val expectedNetReturn: Double,
    @SerialName("expected_volatility") val expectedVolatility: Double,
    @SerialName("funding_cost") val fundingCost: Double,
    val turnover: Double,
    @SerialName("one_day_stress_loss") val oneDayStressLoss: Double,
    @SerialName("multi_session_stress_loss") val multiSessionStressLoss: Double,
    @SerialName("evidence_grade") val evidenceGrade: String,
    val freshness: FreshnessV2,
    val warnings: List<String> = emptyList(),
)

@Serializable
data class BaselinePolicyV2(
    @SerialName("policy_version") val policyVersion: String,
    val weights: List<WeightV2>,
)

@Serializable
data class CanonicalRecommendationBundleV2(
    @SerialName("schema_version") val schemaVersion: Int = 2,
    val session: String,
    @SerialName("as_of") val asOf: String,
    @SerialName("execution_at") val executionAt: String,
    @SerialName("data_version") val dataVersion: String,
    @SerialName("artifact_version") val artifactVersion: String,
    @SerialName("policy_version") val policyVersion: String,
    @SerialName("baseline_policy") val baselinePolicy: BaselinePolicyV2,
    @SerialName("default_risk_profile") val defaultRiskProfile: RiskProfileV2,
    @SerialName("base_recommendation") val baseRecommendation: BaseRecommendationV2,
    @SerialName("default_recommendation") val defaultRecommendation: PortfolioRecommendationV2,
    val disclaimer: String,
)

@Serializable
data class RecommendationPreviewRequestV2(
    val profile: RiskProfileV2,
    @SerialName("risk_state") val riskState: RiskStateV2? = null,
    @SerialName("equity_override") val equityOverride: Double? = null,
    @SerialName("reset_requested") val resetRequested: Boolean = false,
)

// --- staged, loss-aversion-first sniper shadow plan ---

@Serializable
data class SniperPolicyItemV2(
    val rank: Int,
    @SerialName("strategy_id") val strategyId: String,
    val stage: Int,
    val role: String,
    val activation: String,
    val conviction: String,
    val rule: String,
    val reason: String,
)

@Serializable
data class SniperSizingV2(
    @SerialName("account_equity") val accountEquity: Double,
    @SerialName("max_tolerable_loss") val maxTolerableLoss: Double,
    @SerialName("adverse_move_p05") val adverseMoveP05: Double,
    @SerialName("risk_notional") val riskNotional: Double,
    @SerialName("capped_notional") val cappedNotional: Double,
    @SerialName("portfolio_weight") val portfolioWeight: Double,
    @SerialName("estimated_shares") val estimatedShares: Int,
    @SerialName("cap_applied") val capApplied: Boolean,
    @SerialName("resolved_signal_outcomes") val resolvedSignalOutcomes: Int,
    @SerialName("estimate_source") val estimateSource: String,
    val warning: String,
)

@Serializable
data class SniperOutcomeEvidenceV2(
    val observations: Int,
    @SerialName("effective_sample_size") val effectiveSampleSize: Double,
    @SerialName("cost_adjusted_win_rate") val costAdjustedWinRate: Double? = null,
    @SerialName("mean_net_return") val meanNetReturn: Double? = null,
    @SerialName("adverse_move_p05") val adverseMoveP05: Double? = null,
    @SerialName("evidence_grade") val evidenceGrade: String,
    val promoted: Boolean = false,
    val warning: String,
)

@Serializable
data class SniperCandidateV2(
    @SerialName("strategy_id") val strategyId: String,
    @SerialName("component_triggers") val componentTriggers: List<String> = emptyList(),
    val symbol: String,
    val status: String,
    @SerialName("signal_session") val signalSession: String? = null,
    @SerialName("entry_window") val entryWindow: String? = null,
    @SerialName("exit_rule") val exitRule: String,
    @SerialName("maximum_holding_sessions") val maximumHoldingSessions: Int,
    val sizing: SniperSizingV2? = null,
    val evidence: SniperOutcomeEvidenceV2,
    @SerialName("veto_reasons") val vetoReasons: List<String> = emptyList(),
    val cautions: List<String> = emptyList(),
    val actionable: Boolean = false,
    @SerialName("paper_only") val paperOnly: Boolean = true,
)

@Serializable
data class SniperOverlayV2(
    @SerialName("strategy_id") val strategyId: String,
    val state: String,
    val value: Double? = null,
    val threshold: String,
    @SerialName("can_initiate") val canInitiate: Boolean = false,
    val effect: String,
    val warning: String? = null,
)

@Serializable
data class SniperPlanV2(
    @SerialName("schema_version") val schemaVersion: Int = 2,
    @SerialName("generated_at") val generatedAt: String,
    val session: String,
    @SerialName("data_version") val dataVersion: String,
    @SerialName("artifact_version") val artifactVersion: String,
    @SerialName("policy_version") val policyVersion: String,
    @SerialName("account_equity") val accountEquity: Double,
    @SerialName("max_tolerable_loss") val maxTolerableLoss: Double,
    @SerialName("requested_vehicle") val requestedVehicle: String,
    @SerialName("policy_ranking") val policyRanking: List<SniperPolicyItemV2>,
    @SerialName("stage_1_candidates") val stage1Candidates: List<SniperCandidateV2>,
    @SerialName("stage_2_candidates") val stage2Candidates: List<SniperCandidateV2>,
    val overlays: List<SniperOverlayV2>,
    @SerialName("excluded_strategy_ids") val excludedStrategyIds: List<String>,
    @SerialName("stage_1_promotion_satisfied") val stage1PromotionSatisfied: Boolean = false,
    val warnings: List<String> = emptyList(),
    val disclaimer: String,
)

@Serializable
data class SniperPreviewRequestV2(
    @SerialName("account_equity") val accountEquity: Double = 100_000.0,
    @SerialName("max_tolerable_loss") val maxTolerableLoss: Double = 250.0,
    val vehicle: String = "SPY",
)

// --- user-selected instrument analysis; server evidence only ---

@Serializable
data class InstrumentResolutionV2(
    @SerialName("requested_symbol") val requestedSymbol: String,
    @SerialName("resolved_symbol") val resolvedSymbol: String,
    @SerialName("instrument_kind") val instrumentKind: String,
    @SerialName("proxy_for") val proxyFor: String? = null,
    @SerialName("tradeable_instrument") val tradeableInstrument: Boolean = true,
    val notes: List<String> = emptyList(),
)

@Serializable
data class EdgeEffectV2(
    @SerialName("edge_id") val edgeId: String,
    val family: String,
    @SerialName("horizon_sessions") val horizonSessions: Int,
    val direction: String,
    val observation: String,
    @SerialName("positive_contribution") val positiveContribution: Double = 0.0,
    @SerialName("negative_contribution") val negativeContribution: Double = 0.0,
    @SerialName("net_contribution") val netContribution: Double = 0.0,
    @SerialName("protective_avoidance_value") val protectiveAvoidanceValue: Double = 0.0,
    @SerialName("lower_95") val lower95: Double? = null,
    @SerialName("adverse_counter_effect") val adverseCounterEffect: String,
    @SerialName("protective_counter_effect") val protectiveCounterEffect: String,
    @SerialName("evidence_grade") val evidenceGrade: String,
    val compound: Boolean = false,
    val promoted: Boolean = false,
    @SerialName("artifact_hash") val artifactHash: String? = null,
    val invalidation: String,
)

@Serializable
data class TimingWindowV2(
    val horizon: String,
    val label: String,
    @SerialName("entry_window") val entryWindow: String,
    @SerialName("exit_window") val exitWindow: String,
    @SerialName("holding_sessions") val holdingSessions: Int,
    @SerialName("expected_net_return") val expectedNetReturn: Double? = null,
    @SerialName("lower_95") val lower95: Double? = null,
    @SerialName("upper_95") val upper95: Double? = null,
    @SerialName("multiple_testing_adjusted_pvalue")
    val multipleTestingAdjustedPvalue: Double? = null,
    val observations: Int = 0,
    @SerialName("effective_sample_size") val effectiveSampleSize: Double = 0.0,
    @SerialName("evidence_grade") val evidenceGrade: String,
    val actionable: Boolean = false,
    @SerialName("artifact_hash") val artifactHash: String? = null,
    val rationale: String,
    @SerialName("what_invalidates_it") val whatInvalidatesIt: List<String> = emptyList(),
)

@Serializable
data class WinScoreV2(
    @SerialName("net_win_rate") val netWinRate: Double,
    @SerialName("shrunk_win_rate") val shrunkWinRate: Double,
    @SerialName("win_score") val winScore: Double,
    @SerialName("expected_net_return") val expectedNetReturn: Double,
    @SerialName("lower_95") val lower95: Double,
    val observations: Int,
    @SerialName("effective_sample_size") val effectiveSampleSize: Double,
    val rank: Int,
    @SerialName("candidates_ranked") val candidatesRanked: Int,
    @SerialName("multiple_testing_adjusted_pvalue")
    val multipleTestingAdjustedPvalue: Double,
    @SerialName("evidence_grade") val evidenceGrade: String,
    val actionable: Boolean = false,
    val explanation: String = "",
)

@Serializable
data class TailwindCalendarCellV2(
    @SerialName("slot_key") val slotKey: String,
    @SerialName("display_label") val displayLabel: String,
    @SerialName("entry_window") val entryWindow: String,
    @SerialName("exit_window") val exitWindow: String,
    val horizon: String,
    val score: WinScoreV2,
    @SerialName("rank_percentile") val rankPercentile: Double,
)

@Serializable
data class TailwindCalendarV2(
    val resolution: String,
    val timezone: String,
    val horizon: String,
    @SerialName("data_start") val dataStart: String? = null,
    @SerialName("data_end") val dataEnd: String? = null,
    val cells: List<TailwindCalendarCellV2> = emptyList(),
    val warning: String,
)

@Serializable
data class ChosenTimeRatingV2(
    val resolution: String,
    val horizon: String,
    @SerialName("requested_time") val requestedTime: String,
    @SerialName("matched_slot") val matchedSlot: String? = null,
    val rating: String,
    val score: WinScoreV2? = null,
    @SerialName("better_alternative") val betterAlternative: TailwindCalendarCellV2? = null,
    @SerialName("score_improvement") val scoreImprovement: Double? = null,
    val recommendation: String,
    val actionable: Boolean = false,
)

@Serializable
data class ExitPlanV2(
    val horizon: String,
    @SerialName("entry_slot") val entrySlot: String,
    @SerialName("preferred_exit") val preferredExit: String? = null,
    @SerialName("holding_sessions") val holdingSessions: Int,
    @SerialName("data_resolution") val dataResolution: String,
    val score: WinScoreV2? = null,
    val alternatives: List<TailwindCalendarCellV2> = emptyList(),
    val actionable: Boolean = false,
    val rationale: String,
    val warning: String? = null,
)

@Serializable
data class RecheckPlanV2(
    val enabled: Boolean,
    @SerialName("intended_entry_at") val intendedEntryAt: String? = null,
    @SerialName("next_check_at") val nextCheckAt: String? = null,
    @SerialName("cadence_minutes") val cadenceMinutes: Int? = null,
    @SerialName("required_resolution") val requiredResolution: String? = null,
    @SerialName("automatic_recheck_supported") val automaticRecheckSupported: Boolean = true,
    val reason: String,
)

@Serializable
data class HorizonTimingAnalysisV2(
    val horizon: String,
    @SerialName("data_resolution") val dataResolution: String,
    @SerialName("best_window") val bestWindow: TimingWindowV2? = null,
    @SerialName("worst_window") val worstWindow: TimingWindowV2? = null,
    val alternatives: List<TimingWindowV2> = emptyList(),
    @SerialName("intended_entry_assessment") val intendedEntryAssessment: String? = null,
    val actionable: Boolean = false,
    @SerialName("searched_variants") val searchedVariants: Int = 0,
    val warning: String? = null,
)

@Serializable
data class NewsEvidenceV2(
    @SerialName("news_id") val newsId: String,
    val symbol: String,
    val headline: String,
    val source: String,
    @SerialName("published_at") val publishedAt: String,
    val url: String? = null,
    val summary: String = "",
    @SerialName("sentiment_label") val sentimentLabel: String = "UNSCORED",
    val relevance: Double? = null,
    @SerialName("is_fresh") val isFresh: Boolean,
    @SerialName("age_hours") val ageHours: Double,
    val warning: String,
)

@Serializable
data class AlignmentSummaryV2(
    @SerialName("aligned_trade") val alignedTrade: Boolean,
    @SerialName("promoted_tailwinds") val promotedTailwinds: Int,
    @SerialName("promoted_headwinds") val promotedHeadwinds: Int,
    @SerialName("observational_tailwinds") val observationalTailwinds: Int,
    @SerialName("observational_headwinds") val observationalHeadwinds: Int,
    @SerialName("actionable_horizons") val actionableHorizons: List<String> = emptyList(),
    @SerialName("missing_inputs") val missingInputs: List<String> = emptyList(),
    val explanation: String,
)

@Serializable
data class InstrumentAnalysisV2(
    @SerialName("schema_version") val schemaVersion: Int = 2,
    @SerialName("analysis_id") val analysisId: String,
    val resolution: InstrumentResolutionV2,
    @SerialName("as_of") val asOf: String,
    @SerialName("intended_entry_at") val intendedEntryAt: String? = null,
    @SerialName("data_version") val dataVersion: String,
    @SerialName("artifact_version") val artifactVersion: String,
    @SerialName("policy_version") val policyVersion: String,
    @SerialName("current_price") val currentPrice: Double? = null,
    val status: String,
    @SerialName("overall_rating") val overallRating: String,
    @SerialName("overall_score") val overallScore: Double? = null,
    @SerialName("canonical_portfolio_weight") val canonicalPortfolioWeight: Double = 0.0,
    val alignment: AlignmentSummaryV2,
    @SerialName("horizon_analyses") val horizonAnalyses: List<HorizonTimingAnalysisV2>,
    @SerialName("chosen_time_ratings") val chosenTimeRatings: List<ChosenTimeRatingV2> = emptyList(),
    @SerialName("exit_plans") val exitPlans: List<ExitPlanV2> = emptyList(),
    @SerialName("tailwind_calendars") val tailwindCalendars: List<TailwindCalendarV2> = emptyList(),
    @SerialName("recheck_plan")
    val recheckPlan: RecheckPlanV2 = RecheckPlanV2(
        enabled = false,
        reason = "No intended entry time was supplied.",
    ),
    val tailwinds: List<EdgeEffectV2> = emptyList(),
    val headwinds: List<EdgeEffectV2> = emptyList(),
    @SerialName("mixed_effects") val mixedEffects: List<EdgeEffectV2> = emptyList(),
    val news: List<NewsEvidenceV2> = emptyList(),
    @SerialName("what_to_watch") val whatToWatch: List<String> = emptyList(),
    @SerialName("current_year_notes") val currentYearNotes: List<String> = emptyList(),
    val warnings: List<String> = emptyList(),
    val disclaimer: String,
)

@Serializable
data class InstrumentAnalysisRequestV2(
    val symbol: String,
    @SerialName("instrument_kind") val instrumentKind: String? = null,
    @SerialName("intended_entry_at") val intendedEntryAt: String? = null,
    @SerialName("intended_entry_date") val intendedEntryDate: String? = null,
    @SerialName("round_trip_cost_bps") val roundTripCostBps: Double = 10.0,
    @SerialName("include_news") val includeNews: Boolean = true,
)

@Serializable
data class InstrumentRecheckRequestV2(
    @SerialName("previous_analysis") val previousAnalysis: InstrumentAnalysisV2,
    val request: InstrumentAnalysisRequestV2,
)

@Serializable
data class InstrumentRecheckV2(
    @SerialName("previous_analysis_id") val previousAnalysisId: String,
    val analysis: InstrumentAnalysisV2,
    @SerialName("recommendation_still_holds") val recommendationStillHolds: Boolean,
    @SerialName("better_alternative_emerged") val betterAlternativeEmerged: Boolean,
    val changes: List<String> = emptyList(),
)

@Serializable
data class PatternLeaderRequestV2(
    val symbols: List<String>,
    val resolution: String = "DAY",
    val horizon: String = "WEEK",
    val limit: Int = 10,
)

@Serializable
data class PatternLeaderV2(
    val rank: Int,
    val symbol: String,
    val resolution: String,
    val horizon: String,
    @SerialName("strongest_slot") val strongestSlot: TailwindCalendarCellV2,
    val actionable: Boolean = false,
)

@Serializable
data class PatternLeaderBoardV2(
    @SerialName("as_of") val asOf: String,
    val resolution: String,
    val horizon: String,
    @SerialName("searched_symbols") val searchedSymbols: List<String>,
    @SerialName("skipped_symbols") val skippedSymbols: List<String> = emptyList(),
    @SerialName("searched_cells") val searchedCells: Int,
    val leaders: List<PatternLeaderV2>,
    val warning: String,
)

/** One scheduled macro release from the bundled seed (scripts/export_macro_events.py). */
@Serializable
data class MacroEvent(
    val date: String,                       // ISO yyyy-MM-dd
    val type: String,                       // FOMC | CPI | EIA
    val label: String,
    @SerialName("time_et") val timeEt: String,
)

@Serializable
data class MacroEvents(
    @SerialName("schema_version") val schemaVersion: Int = 1,
    @SerialName("generated_at") val generatedAt: String = "",
    val events: List<MacroEvent> = emptyList(),
)

/** One row of GET /edges — the validated edge catalog with lifecycle status. */
@Serializable
data class EdgeSummaryV2(
    @SerialName("edge_id") val edgeId: String,
    val name: String = "",
    val family: String = "",
    val direction: String = "",
    val horizon: Int = 0,
    val status: String = "",
    @SerialName("net_mean_return") val netMeanReturn: Double? = null,
    @SerialName("q_value") val qValue: Double? = null,
    @SerialName("deflated_sharpe") val deflatedSharpe: Double? = null,
    @SerialName("sample_size") val sampleSize: Int? = null,
)

@Serializable
data class EdgeCatalog(val edges: List<EdgeSummaryV2> = emptyList())

/** One row of GET /backtests. */
@Serializable
data class BacktestRunV2(
    @SerialName("run_id") val runId: String,
    @SerialName("created_at") val createdAt: String = "",
    @SerialName("cost_scenario") val costScenario: String = "",
)
