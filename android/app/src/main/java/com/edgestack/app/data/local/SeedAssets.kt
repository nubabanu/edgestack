package com.edgestack.app.data.local

import android.content.Context
import com.edgestack.app.core.AppJson
import com.edgestack.app.domain.model.CalendarBundle
import com.edgestack.app.domain.model.CanonicalRecommendationBundleV2

/** Bundled seed data under assets/seed, exported by scripts/export_mobile_bundle.py. */
class SeedAssets(private val context: Context) {

    private fun read(name: String): String =
        context.assets.open("seed/$name").bufferedReader().readText()

    fun calendar(): CalendarBundle =
        AppJson.decodeFromString(CalendarBundle.serializer(), read("calendar.json"))

    fun recommendation(): CanonicalRecommendationBundleV2 =
        AppJson.decodeFromString(
            CanonicalRecommendationBundleV2.serializer(),
            read("recommendation.json"),
        )
}
