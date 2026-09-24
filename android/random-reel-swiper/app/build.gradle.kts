plugins {
    id("com.android.application")
    testOptions { unitTests.isIncludeAndroidResources = false }
}


android {
    namespace = "com.airi.randomreelswiper"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.airi.randomreelswiper"
        minSdk = 24
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

dependencies {
    testImplementation("junit:junit:4.13.2")
}
