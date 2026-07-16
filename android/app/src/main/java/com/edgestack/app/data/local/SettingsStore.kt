package com.edgestack.app.data.local

import android.content.Context
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.doublePreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.longPreferencesKey
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.map

private val Context.dataStore by preferencesDataStore(name = "settings")

data class Settings(
    val baseUrl: String = "",
    val baseLeverage: Double = 1.0,
    val alertsBuyAtClose: Boolean = true,
    val alertsTurnOfMonth: Boolean = true,
    val alertsSeptember: Boolean = true,
    val alertsFebSniper: Boolean = true,
    val alertsNovWorstDay: Boolean = true,
    val lastSyncEpochMs: Long = 0L,
    val lastAbove200Dma: Boolean? = null,
    val lastVolGateOn: Boolean? = null,
)

class SettingsStore(private val context: Context) {

    private object Keys {
        val baseUrl = stringPreferencesKey("base_url")
        val baseLeverage = doublePreferencesKey("base_leverage")
        val buyAtClose = booleanPreferencesKey("alerts_buy_at_close")
        val turnOfMonth = booleanPreferencesKey("alerts_tom")
        val september = booleanPreferencesKey("alerts_september")
        val febSniper = booleanPreferencesKey("alerts_feb")
        val novWorstDay = booleanPreferencesKey("alerts_nov")
        val lastSync = longPreferencesKey("last_sync_ms")
        val above200 = booleanPreferencesKey("last_above_200dma")
        val volGate = booleanPreferencesKey("last_vol_gate_on")
    }

    val settings: Flow<Settings> = context.dataStore.data.map { p ->
        Settings(
            baseUrl = p[Keys.baseUrl] ?: "",
            baseLeverage = p[Keys.baseLeverage] ?: 1.0,
            alertsBuyAtClose = p[Keys.buyAtClose] ?: true,
            alertsTurnOfMonth = p[Keys.turnOfMonth] ?: true,
            alertsSeptember = p[Keys.september] ?: true,
            alertsFebSniper = p[Keys.febSniper] ?: true,
            alertsNovWorstDay = p[Keys.novWorstDay] ?: true,
            lastSyncEpochMs = p[Keys.lastSync] ?: 0L,
            lastAbove200Dma = p[Keys.above200],
            lastVolGateOn = p[Keys.volGate],
        )
    }

    suspend fun current(): Settings = settings.first()

    suspend fun setBaseUrl(v: String) =
        context.dataStore.edit { it[Keys.baseUrl] = v }

    suspend fun setBaseLeverage(v: Double) =
        context.dataStore.edit { it[Keys.baseLeverage] = v }

    suspend fun setAlert(type: String, enabled: Boolean) =
        context.dataStore.edit {
            when (type) {
                "buyAtClose" -> it[Keys.buyAtClose] = enabled
                "turnOfMonth" -> it[Keys.turnOfMonth] = enabled
                "september" -> it[Keys.september] = enabled
                "febSniper" -> it[Keys.febSniper] = enabled
                "novWorstDay" -> it[Keys.novWorstDay] = enabled
            }
        }

    suspend fun stampSync(epochMs: Long) =
        context.dataStore.edit { it[Keys.lastSync] = epochMs }

    suspend fun setRiskState(above200: Boolean, volGateOn: Boolean) =
        context.dataStore.edit {
            it[Keys.above200] = above200
            it[Keys.volGate] = volGateOn
        }
}
