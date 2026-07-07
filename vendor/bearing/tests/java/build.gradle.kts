plugins {
    java
    application
}

repositories {
    mavenCentral()
}

dependencies {
    implementation("org.apache.lucene:lucene-core:10.3.2")
    implementation("org.apache.lucene:lucene-queryparser:10.3.2")
    implementation("com.google.code.gson:gson:2.11.0")
}

java {
    toolchain {
        languageVersion = JavaLanguageVersion.of(21)
    }
}

tasks.register<JavaExec>("verifyIndex") {
    classpath = sourceSets["main"].runtimeClasspath
    mainClass = "VerifyIndex"
    val indexDir = project.findProperty("indexDir") as? String
    val docCount = project.findProperty("docCount") as? String
    args = listOfNotNull(indexDir, docCount)
}

tasks.register<JavaExec>("verifyImpacts") {
    classpath = sourceSets["main"].runtimeClasspath
    mainClass = "VerifyImpacts"
    val indexDir = project.findProperty("indexDir") as? String
    args = listOfNotNull(indexDir)
}


tasks.register<JavaExec>("indexAllFields") {
    classpath = sourceSets["main"].runtimeClasspath
    mainClass = "IndexAllFields"
    val docsDir = project.findProperty("docsDir") as? String
    val indexDir = project.findProperty("indexDir") as? String
    val threads = project.findProperty("threads") as? String
    val compound = project.findProperty("compound") as? String
    args = listOfNotNull(docsDir, indexDir) +
        (if (threads != null) listOf("--threads", threads) else emptyList()) +
        (if (compound != null) listOf("--compound") else emptyList())
}


tasks.register<JavaExec>("generateIndexSummary") {
    classpath = sourceSets["main"].runtimeClasspath
    mainClass = "GenerateIndexSummary"
    val indexDir = project.findProperty("indexDir") as? String
    args = listOfNotNull(indexDir)
}

tasks.register<JavaExec>("queryIndex") {
    classpath = sourceSets["main"].runtimeClasspath
    mainClass = "QueryIndex"
    val indexDir = project.findProperty("indexDir") as? String
    val queriesFile = project.findProperty("queriesFile") as? String
    val outputFile = project.findProperty("outputFile") as? String
    args = listOfNotNull(indexDir, queriesFile, outputFile)
}

// Generic task: run any main class without registering a dedicated task.
//   ./tests/java/gradlew -p tests/java -q runJava -PmainClass=MyTool -Pargs="arg1 arg2 arg3"
tasks.register<JavaExec>("runJava") {
    classpath = sourceSets["main"].runtimeClasspath
    mainClass = (project.findProperty("mainClass") as? String) ?: "UNSET"
    val rawArgs = project.findProperty("args") as? String ?: ""
    args = if (rawArgs.isBlank()) emptyList() else rawArgs.split(" ")
}
