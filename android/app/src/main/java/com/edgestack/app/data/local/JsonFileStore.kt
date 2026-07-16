package com.edgestack.app.data.local

import android.content.Context
import kotlinx.serialization.KSerializer
import kotlinx.serialization.json.Json
import java.io.File

/** Tiny whole-document JSON cache in filesDir/cache-json/ (atomic writes). */
class JsonFileStore(context: Context, private val json: Json) {

    private val dir: File = File(context.filesDir, "cache-json").apply { mkdirs() }

    fun <T> read(name: String, serializer: KSerializer<T>): T? {
        val f = File(dir, name)
        if (!f.exists()) return null
        return runCatching {
            json.decodeFromString(serializer, f.readText())
        }.getOrNull()
    }

    fun <T> write(name: String, serializer: KSerializer<T>, value: T) {
        val tmp = File(dir, "$name.tmp")
        tmp.writeText(json.encodeToString(serializer, value))
        if (!tmp.renameTo(File(dir, name))) {
            File(dir, name).delete()
            tmp.renameTo(File(dir, name))
        }
    }
}
