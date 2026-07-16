package com.edgestack.app.data.local

import android.content.Context
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.doublePreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.longPreferencesKey
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import com.edgestack.app.core.AppJson
import com.edgestack.app.domain.model.RiskProfileV2
import com.edgestack.app.domain.model.RiskStateV2
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.map

private val Context.dataStore by preferencesDataStore(name = "settings")

data class Settings(
    val baseUrl: String = "",
    val targetVolatility: Double = 0.12,
    val maximumDrawdown: Double = 0.15,
    val maximumGrossLeverage: Double = 1.0,
    val fundingSpreadBps: Double = 200.0,
    val perStockCap: Double = 0.03,
    val sectorCap: Double = 0.20,
    val lastSyncEpochMs: Long = 0L,
    val lastCanonicalStatus: String = "",
    val lastCanonicalFingerprint: String = "",
    val lastFresh: Boolean? = null,
    val lastRiskStateJson: String = "",
) {
    fun profile(accountEquity: Double = 100_000.0) = RiskProfileV2(
        accountEquity = accountEquity,
        targetVolatility = targetVolatility,
        maximumDrawdown = maximumDrawdown,
        maximumGrossLeverage = maximumGrossLeverage,
        fundingSpreadBps = fundingSpreadBps,
        perStockCap = perStockCap,
        sectorCap = sectorCap,
    )
}

class SettingsStore(private val context: Context) {

    private object Keys {
        val baseUrl = stringPreferencesKey("base_url")
        val legacyBaseLeverage = doublePreferencesKey("base_leverage")
        val leverageMigrated = booleanPreferencesKey("risk_v2_leverage_migrated")
        val targetVolatility = doublePreferencesKey("risk_v2_target_volatility")
        val maximumDrawdown = doublePreferencesKey("risk_v2_maximum_drawdown")
        val maximumGrossLeverage = doublePreferencesKey("risk_v2_maximum_gross_leverage")
        val fundingSpreadBps = doublePreferencesKey("risk_v2_funding_spread_bps")
        val perStockCap = doublePreferencesKey("risk_v2_per_stock_cap")
        val sectorCap = doublePreferencesKey("risk_v2_sector_cap")
        val lastSync = longPreferencesKey("last_sync_ms")
        val lastStatus = stringPreferencesKey("canonical_last_status")
        val lastFingerprint = stringPreferencesKey("canonical_last_fingerprint")
        val lastFresh = booleanPreferencesKey("canonical_last_fresh")
        val lastRiskState = stringPreferencesKey("canonical_risk_state_json")
    }

    val settings: Flow<Settings> = context.dataStore.data.map { p ->
        Settings(
            baseUrl = p[Keys.baseUrl] ?: "",
            targetVolatility = p[Keys.targetVolatility] ?: 0.12,
            maximumDrawdown = p[Keys.maximumDrawdown] ?: 0.15,
            maximumGrossLeverage = p[Keys.maximumGrossLeverage]
                ?: p[Keys.legacyBaseLeverage]?.coerceIn(0.0, 5.0) ?: 1.0,
            fundingSpreadBps = p[Keys.fundingSpreadBps] ?: 200.0,
            perStockCap = p[Keys.perStockCap] ?: 0.03,
            sectorCap = p[Keys.sectorCap] ?: 0.20,
            lastSyncEpochMs = p[Keys.lastSync] ?: 0L,
            lastCanonicalStatus = p[Keys.lastStatus] ?: "",
            lastCanonicalFingerprint = p[Keys.lastFingerprint] ?: "",
            lastFresh = p[Keys.lastFresh],
            lastRiskStateJson = p[Keys.lastRiskState] ?: "",
        )
    }

    suspend fun current(): Settings {
        migrateLegacyLeverageOnce()
        return settings.first()
    }

    private suspend fun migrateLegacyLeverageOnce() = context.dataStore.edit { p ->
        if (p[Keys.leverageMigrated] != true) {
            if (p[Keys.maximumGrossLeverage] == null) {
                p[Keys.maximumGrossLeverage] =
                    (p[Keys.legacyBaseLeverage] ?: 1.0).coerceIn(0.0, 5.0)
            }
            p[Keys.leverageMigrated] = true
        }
    }

    suspend fun setBaseUrl(value: String) = context.dataStore.edit { it[Keys.baseUrl] = value }

    suspend fun setRiskProfile(profile: RiskProfileV2) = context.dataStore.edit {
        it[Keys.targetVolatility] = profile.targetVolatility.coerceIn(0.0001, 0.30)
        it[Keys.maximumDrawdown] = profile.maximumDrawdown.coerceIn(0.0001, 0.50)
        it[Keys.maximumGrossLeverage] = profile.maximumGrossLeverage.coerceIn(0.0, 5.0)
        it[Keys.fundingSpreadBps] = profile.fundingSpreadBps.coerceIn(0.0, 2_000.0)
        it[Keys.perStockCap] = profile.perStockCap.coerceIn(0.0001, 0.10)
        it[Keys.sectorCap] = profile.sectorCap.coerceIn(0.0001, 1.0)
    }

    suspend fun stampSync(epochMs: Long) = context.dataStore.edit { it[Keys.lastSync] = epochMs }

    fun decodedRiskState(settings: Settings): RiskStateV2? =
        settings.lastRiskStateJson.takeIf { it.isNotBlank() }?.let {
            runCatching { AppJson.decodeFromString(RiskStateV2.serializer(), it) }.getOrNull()
        }

    suspend fun saveCanonicalState(
        status: String,
        fingerprint: String,
        fresh: Boolean,
        riskState: RiskStateV2,
    ) = context.dataStore.edit {
        it[Keys.lastStatus] = status
        it[Keys.lastFingerprint] = fingerprint
        it[Keys.lastFresh] = fresh
        it[Keys.lastRiskState] = AppJson.encodeToString(RiskStateV2.serializer(), riskState)
    }
}
