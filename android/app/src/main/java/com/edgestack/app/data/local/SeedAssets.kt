package com.edgestack.app.data.local

import android.content.Context
import com.edgestack.app.core.AppJson
import com.edgestack.app.domain.model.Board
import com.edgestack.app.domain.model.CalendarBundle
import com.edgestack.app.domain.model.EdgesBundle

/** Bundled seed data under assets/seed, exported by scripts/export_mobile_bundle.py. */
class SeedAssets(private val context: Context) {

    private fun read(name: String): String =
        context.assets.open("seed/$name").bufferedReader().readText()

    fun board(): Board =
        AppJson.decodeFromString(Board.serializer(), read("board.json"))

    fun edges(): EdgesBundle =
        AppJson.decodeFromString(EdgesBundle.serializer(), read("edges.json"))

    fun calendar(): CalendarBundle =
        AppJson.decodeFromString(CalendarBundle.serializer(), read("calendar.json"))
}
