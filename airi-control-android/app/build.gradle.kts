plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

val airiControlRelay = providers.gradleProperty("airiControlRelay").orElse("").get()
val airiControlTopic = providers.gradleProperty("airiControlTopic").orElse("").get()
fun quotedBuildConfig(value: String): String =
    """ + value.replace("\", "\\").replace(""", "\"") + """

android {
    namespace = "com.airipc.control"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.airipc.control"
        minSdk = 26
        targetSdk = 35
        versionCode = 2
        versionName = "0.2.0"
        buildConfigField("String", "CONTROL_RELAY_OVERRIDE", quotedBuildConfig(airiControlRelay))
        buildConfigField("String", "CONTROL_TOPIC_OVERRIDE", quotedBuildConfig(airiControlTopic))
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    buildFeatures {
        compose = true
        buildConfig = true
    }

    packaging {
        resources.excludes += "/META-INF/{AL2.0,LGPL2.1}"
    }
}

kotlin {
    jvmToolchain(17)
}

dependencies {
    implementation(platform("androidx.compose:compose-bom:2025.05.01"))
    implementation("androidx.activity:activity-compose:1.10.1")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.9.0")
    implementation("androidx.core:core-ktx:1.16.0")
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.10.1")
    debugImplementation("androidx.compose.ui:ui-tooling")
}
