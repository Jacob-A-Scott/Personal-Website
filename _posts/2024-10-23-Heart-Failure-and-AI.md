---
layout: post
title: "Heart Failure & AI: Predicting Outcomes with Deep Learning"
description: "Describing my master's capstone project, where I'm utilizing deep learning and multi-modal data to predict readmissions for ICU heart failure patients."
date:   2024-10-23
image:  '/images/bedside_monitor.jpg'
tags:
  - "ai"
  - "machine learning"
  - "deep learning"
  - "transformers"
---

## Setting the Stage
In recent years, the healthcare landscape has been revolutionized by advancements in data-driven insights. From diagnostics to patient monitoring, AI domains such as data mining, machine learning, and deep learning have started to shape a new frontier for predicting health outcomes. To fulfill my capstone requirement for my master’s degree in Data Science this semester, I chose to take on an independent research project and explore this intersection of technology and healthcare, with a focus on leveraging machine learning to improve outcomes for patients with heart failure. This post will give you a peek into the project—what I’m aiming to do, why it's significant, and how I’m tackling it.

Heart failure is a complex, chronic condition, and its management is challenging for healthcare providers and patients alike. The need for hospitalization, often sudden and unpredictable, takes a toll on both the healthcare system and patient quality of life. My capstone project aims to build a predictive model for forecasting 30d hospital readmissions for heart failure ICU patients using the MIMIC-IV<sup id="citation1"><a href="#ref1"> [1]</a></sup> database. By accurately predicting these outcomes, the intent is to empower healthcare professionals with early warning signals that could guide preventive care measures, ultimately improving patient outcomes. I’m under no illusion that this work could be utilized immediately in practice, but contributing to this area by exploring my intended multi-modal approach is my objective for this research.

## Multi-Modal Data for a Multi-Faceted Problem
At its core, this project uses multiple types of data to create a holistic picture of patient health. The dataset has been carefully segmented into three primary components: 
1. **Static Patient-Level Data:** Information like demographics and existing comorbidities. These are the static, foundational details of a patient that stay constant.
2. **Time-Series Vitals:** ICU bedside monitoring data that provides a dense, ongoing record of vital signs. This captures the more dynamic aspects of a patient's health that can change by the hour or by the minute.
3. **Sparse Event Data:** Information like lab results and medication administrations, which occur irregularly but are crucial to understanding patient state shifts.

These different types of data are naturally complementary—each reveals an important aspect of patient health at varying resolutions. My goal is to blend these different data types into a unified model capable of making nuanced predictions about patient outcomes. 

## Why Transformers?
A key challenge in this project is figuring out just how to integrate these diverse data types effectively. Traditional machine learning would require concatenating or otherwise integrating these datasets together to use for predictive modeling. In order to overcome this limitation, I'm planning to use a transformer-based architecture. Transformers, originally developed for natural language processing and now ubiquitous in the form of LLMs like ChatGPT and Claude, excel at drawing relationships within data sequences—making them an exciting choice for time series data like vitals and event records. By leveraging transformers, my hope is to combine patient-level static features, dynamic vitals, and sparse event data without compromising their unique qualities.

Transformers are powerful in dealing with contextual dependencies—in this case, understanding how variables such as a patient's heart rate, recent medication, vasopressor administration, and age interact to signal a potential hospital readmission risk. The multi-headed attention mechanism inherent in transformers might offer the right balance to weigh each of these signals appropriately.

![Rest]({{ site.baseurl }}/images/hf_mimic.png)
*A general idea of the multi-modal data I'm incorporating into the model.*

## Impact and Future Direction
The goal of this project extends beyond the creation of a predictive model and fulfilling a degree requirement. I want to contribute something small, yet meaningful to the body of research surrounding chronic care management.

Imagine a clinician receiving an alert that a specific heart failure patient is at a heightened risk of readmission in the next 30 days. Such a warning could prompt a change in post-discharge care—maybe increased monitoring, medication adjustment, an extra follow-up appointment. It’s a glimpse into a future of personalized, data-driven healthcare.

This project builds on an existing corpus of research using MIMIC data, even combining it with deep learning and heart failure objectives. My hope is that I learn as much as I can from this research and create something that can be iterated on in the future, perhaps integrating even more varied data sources or refining the model to ensure it is applicable, ethical, and secure enough for real-world healthcare settings.

Thanks for reading. If all goes well, I will have a research paper to share on the topic in January!

\- Jacob

## References
<ol>
  <li id="ref1">[MIMIC-IV, a freely accessible electronic health record dataset.](https://rdcu.be/dXU5M)<a href="#citation1"> [Back to text]</a></li>
</ol>