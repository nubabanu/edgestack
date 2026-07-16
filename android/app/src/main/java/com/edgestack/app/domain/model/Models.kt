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
