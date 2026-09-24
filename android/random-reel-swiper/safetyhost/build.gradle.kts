plugins {
    id("com.android.application")
}

android {
    namespace = "com.example.safetyhost"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.instagram.android"
        minSdk = 24
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}
