---
layout: post
title: "AI vs. Machine Learning: What's the Difference?"
description: "A brief writeup on the overlap and distinction between two of the buzziest topics of our time."
date:   2023-12-22
image:  '/images/ai_vs_ml.jpg'
tags:
  - "ai"
  - "ml"
  - "history"
---

## Data Science Is a Confusing Word Soup
I had originally planned on my first _new_ post on this site being more of a technical walkthrough of a project that I recently completed for my Natural Language Processing (NLP) coursework. However, after reflecting on the topic and considering that NLP is nestled within the often confusing hierarchy of data science disciplines and domains, I thought it would be a good idea to talk about the "family tree" of intelligent computing disciplines and to try to illuminate some of the gray areas between these fields of study and application.

It's only quite recently that Artificial Intelligence (AI) really left the realms of science fiction and highly specialized research and entered the public zeitgeist in its genuine form. Currently, at the end of 2023, countless products and services blazon the term with seemingly every organization and individual "leveraging the power of AI". Buzzwords in tech catch on quickly and their usage spreads exponentially, often without considerations for what the words actually mean.

## So What Is AI?
### A Bit of History
When you hear the term artificial intelligence, it probably conjures up images of KITT from _Knight Rider_ or of a highly intelligent computer locking you out of your spacecraft to leave you to die in the cold vacuum of space. _Fortunately_, while sci-fi tropes shape our preconceived notions of AI, it in fact has a much more grounded history in computer science, with the term first being coined by John McCarthy of MIT.<sup>\[1\]</sup> McCarthy described the nascent field as "the science and engineering of making intelligent machines".<sup>\[2\]</sup> AI, in its essence, is the concept of programming machines to behave like humans, accomplishing complex tasks in clever ways.
### An Umbrella Term Today
While artificial intelligence still encompasses John McCarthy's definition, the field has been shaped and pursued due to practicality and interdisciplinary influences. McCarthy's original vision of artificial _general_ intelligence (AGI) is still a far-off and rather academic idea that despite what AI chatbot companies may say, does not exist today.

> "I'm sorry Dave, I'm afraid I can't do that"... yet?

What AI encompasses today, is all of the methods in which we make computers "do smart stuff." But that "stuff" is narrow in scope, i.e. Narrow AI or Weak AI. We've gotten very good at leveraging computing to solve specific challenges. We can create models that can detect cancer in an X-ray image, curate very personalized recommendations for what movie you should watch next, and create online chatbots that can collate information and generate sometimes _alarmingly_ coherent and complex responses. What these examples, and much of the overall discipline of artificial intelligence in how its been shaped over the course of the last roughly 70 years, have in common is that they are all _data-driven_ approaches. All three of these examples, and most that you would think of\*, are based on collecting data and feeding it into a model, which then "learns" to produce an output.

\* _Examples of non data-driven approaches to AI aren't likely ones you might think of, considering AI as its commonly interpreted. They aren't particularly "intelligent" in their own right and rely mostly on human encoding. For example, a medical diagnosis algorithm (e.g. fever, cough, and sore throat equals a common cold diagnosis) is a form of artificial intelligence as McCarthy's era of computer scientists defined it, albeit a rudimentary one._

This logically leads us to...

## What Is Machine Learning?
### Origins
Machine learning is the culmination of the early years of AI research, a long "winter" in AI breakthroughs, and both a surge in computational power and in the availability of large quantities of data. 

Machine learning has been around for a few decades, relegated to somewhat niche applications of advanced statistical techniques. Although simpler learning\* techniques like linear regression have been around since the 18th century, it wasn't until the rapid expansion of computing that large-scale calculations were able to be performed automatically. Such automation capability naturally led to the pursuit of more complex algorithms than your standard least-squares regression, and thus we have seen an explosion in both the number of and complexity of machine learning algorithms and models in the past two decades, particularly since the 2010s.

\* _Machine learning essentially boils down to an algorithm processing data to "learn" numeric weights to apply to **new** data in order to generate an output._

> All machine learning is AI, but not all AI is machine learning.

### Chatbots Galore!
Machine learning is a data-driven way to pursue artificial intelligence, although what that means in practice is that we are limited in the intelligence we can develop by the data that we feed to any particular machine learning model. ML takes us beyond pure statistics, taking pieces from the discipline along with others, like optimization mathematics (learning weights), cognitive science (neural network modeling), obviously computer science for feasibility, implementation, and more. 

Even the most complex AI tools we see today, and I'm specifically talking about the large language models that are a sensation at the moment (ChatGPT, Bard, Gemini, etc.) are all very complex machine learning models. Through complicated natural language processing (NLP; subset of ML for linguistics), they are able to simulate intelligence by being fed massive amounts of textual training data. There is debate on whether these models truly "understand" language or that they are merely stochastic parrots, but that debate typically devolves into a debate on what "understand" means. And that is a digression far beyond what I want to get into!

> "It's difficult to be rigorous about whether a machine really 'knows', 'thinks', etc., because we're hard put to define these things. We understand human mental processes only slightly better than a fish understands swimming." <sup>\[3\]</sup>
>
><cite>John McCarthy</cite>

### What I Hope to Cover Here
On this blog, I will generally write about data science, with a specific focus on machine learning. I like digging into the practical applications of ML; finding a problem and solving it or creating a better solution than I otherwise could, with machine learning. It is a meteoric field with a breadth of applications and is continually changing our world and will continue to do so. I study it, work with it in my day job, and generally find it to be endlessly fascinating, and I hope you do too! Otherwise, you might not have read this line.

Thanks for reading!

\- Jacob

## References
1. [Wikipedia: John McCarthy](https://en.wikipedia.org/wiki/John_McCarthy_(computer_scientist))
2. "The Little Thoughts of Thinking Machines", Psychology Today, December 1983, pp. 46–49.
3. [Stanford University: Professor John McCarthy](http://jmc.stanford.edu/artificial-intelligence/what-is-ai/index.html)