# My Task in Touché 2026

My work is positioned inside the Touché @ CLEF 2026 shared task **Advertisement in Retrieval-Augmented Generation**, with a primary focus on **Subtask 1: Ad classification**.

## Core Objective

Subtask 1 asks whether a given RAG response contains advertising content.

Input:

- `response`
- `query`
- auxiliary metadata

Output:

- binary label: advertisement present or not (`0/1`)

This is a response-level classification problem.

## My Project Scope

My project can be described as a **Subtask 1 extension** with the goal of improving advertisement detection beyond the basic shared-task baseline.

The main components of my work are:

- replicate and understand the Touché baseline setting
- improve ad detection with stronger classifier setups
- evaluate different feature representations for advertisement detection
- test whether semantic comparison against neutral rewrites improves classification
- prepare the work so it can be evaluated inside the Touché task setting and, if needed, on additional external conditions

## Research Focus

The central research question of my work is whether the **semantic delta between an original response and a neutral rewrite** provides a useful signal for advertisement detection.

This leads to several concrete directions:

- compare end-to-end fine-tuned classifiers against delta-based models
- test whether query-aware detection improves results
- test whether multiple neutral references help
- analyze whether sentence-level transitions or local semantic changes expose ads more clearly
- investigate whether provider-specific wording or advertiser bias influences classification

## Detection Pipeline

The practical pipeline for my task is:

1. Take a `query` and a generated `response`
2. Optionally generate one or more neutral rewrites that remove persuasive content
3. Extract model inputs or semantic-delta features
4. Run a classifier
5. Predict `ad` or `no ad`

In this repository, that includes both:

- fine-tuned transformer classifiers
- frozen-encoder feature models based on response-vs-neutral differences

## Why This Is a Good Thesis Fit

This task is well suited for a thesis because it combines:

- a current shared-task benchmark
- a societally relevant safety problem in LLM and RAG systems
- a clear supervised evaluation setting
- room for methodological improvement through better features and model design

The project is not only about reproducing a benchmark. It studies whether new signals, especially **semantic delta features**, can improve advertisement detection in generated responses.

## Thesis Positioning

A concise thesis framing is:

> The Touché 2026 shared task "Advertisement in Retrieval-Augmented Generation" investigates how advertisements can be detected and removed from RAG-based responses. My work focuses on Subtask 1, which classifies responses as containing advertisements or not. The project evaluates whether semantic differences between original responses and neutral rewrites can improve advertisement detection beyond standard fine-tuned baselines.

## Expected Deliverables

The expected outputs of this work are:

- a reproducible Subtask 1 classification pipeline
- comparison of multiple model families
- evaluation of semantic-delta features
- analysis of query-aware and multi-neutral setups
- results that can support a Touché-style submission and the written thesis
