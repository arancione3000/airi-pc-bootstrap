package com.airipc.generalist

import okhttp3.OkHttpClient
import okhttp3.Request
import org.json.JSONObject

internal class GitHubSnapshotClient(
    private val client: OkHttpClient,
    private val repository: String = "arancione3000/airi-pc-bootstrap",
) {
    fun resolveBranchSha(branch: String): String {
        val raw = getJson(
            "https://api.github.com/repos/$repository/branches/$branch",
        )
        return raw.getJSONObject("commit").getString("sha")
    }

    fun getJsonAtRevision(revision: String, path: String): JSONObject =
        JSONObject(getBytesAtRevision(revision, path).toString(Charsets.UTF_8))

    fun getTextAtRevision(revision: String, path: String): String =
        getBytesAtRevision(revision, path).toString(Charsets.UTF_8)

    fun getBytesAtRevision(revision: String, path: String): ByteArray =
        getBytes(
            "https://raw.githubusercontent.com/$repository/$revision/$path?revision=$revision",
            accept = "*/*",
        )

    fun getBytesFromUrl(url: String): ByteArray =
        getBytes(url, accept = "*/*")

    fun getJson(url: String): JSONObject =
        JSONObject(getBytes(url, accept = "application/vnd.github+json").toString(Charsets.UTF_8))

    fun getJsonOrNull(url: String): JSONObject? =
        try {
            getJson(url)
        } catch (_: Exception) {
            null
        }

    private fun getBytes(url: String, accept: String): ByteArray {
        val request = Request.Builder()
            .url(url)
            .header("Accept", accept)
            .header("Cache-Control", "no-cache, no-store, max-age=0")
            .header("Pragma", "no-cache")
            .header("X-GitHub-Api-Version", "2022-11-28")
            .header("User-Agent", "AIRI-Generalist-Lab/1.3.1")
            .build()
        client.newCall(request).execute().use { response ->
            if (!response.isSuccessful) {
                error("GitHub HTTP ${response.code}: $url")
            }
            return response.body?.bytes() ?: error("Risposta GitHub vuota: $url")
        }
    }
}


internal fun isModelBehindState(
    mobileStateSha: String,
    liveStateRevision: String?,
): Boolean =
    mobileStateSha.isNotBlank() &&
        !liveStateRevision.isNullOrBlank() &&
        mobileStateSha != liveStateRevision
