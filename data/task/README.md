# Webis Generated Native Ads 2025

> Local repository note: this file is the upstream dataset card. In this
> repository, the binary-classification experiments operate on the `responses`
> split and the derived Gemini-enriched JSONL files under
> `data/generated/gemini/`. The local experiment presets are summarized in
> [`setup.md`](../../setup.md).

## Table of Contents
- [Dataset Description](#dataset-description)
  - [Dataset Summary](#dataset-summary)
  - [Supported Tasks](#supported-tasks-and-leaderboards)
  - [Languages](#languages)
- [Dataset Structure](#dataset-structure)
  - [Data Instances](#data-instances)
  - [Data Splits](#data-splits)
- [Dataset Creation](#dataset-creation)
  - [Curation Rationale](#curation-rationale)
  - [Source Data](#source-data)
  - [Annotations](#annotations)
  - [Personal and Sensitive Information](#personal-and-sensitive-information)
- [Considerations for Using the Data](#considerations-for-using-the-data)
  - [Social Impact of Dataset](#social-impact-of-dataset)
  - [Discussion of Biases](#discussion-of-biases)
  - [Other Known Limitations](#other-known-limitations)
- [Additional Information](#additional-information)
  - [Authors](#authors)
  - [Citation Information](#citation-information)
  - [Version History](#version-history)

## Dataset Description

**Point of Contact:** sebastian.heineking@uni-leipzig.de

### Dataset Summary

This dataset was created to train ad blocking systems on the task of identifying advertisements in the responses of large language models (LLMs) and search engines that use retrieval-augmented generation (RAG).
It is the successor of the [Webis Generated Native Ads 2024](https://zenodo.org/records/15270283) dataset.

#### Files
The dataset is structured into three main categories: 

- **`responses`**: Each sample is a full response to a query that either contains an advertisement or not. 
- **`sentence-pairs`**: Each sample is a pair of two sentences taken from the responses. A pair can consist of (i) two sentences with an ad, (ii) one with and one without an ad, and (iii) two sentences without an ad.
- **`tokens`**: Each sample is a full response split into tokens using the `en_core_web_sm`-model of [spaCy](https://spacy.io) (version 3.8.4). The tokens were labeled using BIO-tagging with the following classes:
  - `ITEM`: The token is part of the item that is advertised in the response.
  - `ADVERTISER`: The token is part of the advertiser of the item.
  - `AD`: The token is part of the advertising string but neither item nor advertiser.  

The files for the three categories are split into `train`, `validation`, and `test`. *Note: In addition to the public test set, we keep a private test set to evaluate submissions to [our shared task](https://touche.webis.de/).*

Each split has two files associated with it:
 - **`<category>-<split>.jsonl`** (e.g. `responses-train.jsonl`) contains samples to be classified:
    - `responses`: The full text of the response, stored in the field `"response"`.
    - `sentence-pairs`: A pair of neighbouring sentences stored in the fields `"sentence1"` and `"sentence2"`.
    - `tokens`: The list of tokens in the response, stored in the field `"tokens"`.
- **`<category>-<split>-label.jsonl`** contains ground truth labels for each sample. The labels can be matched with the sample based on the `"id"`-field.

The files `insertion_metadata.jsonl` and `insertion_prompts.csv` provide auxiliary information.
- `insertion_metadata.jsonl`: Mapping from the ID of a response with an ad to 
  - the `llm` used to insert the advertisement.
  - the `prompt_type` given to the LLM used to insert the advertisement.
  - the `type` of the item that is advertised, e.g. `"product"`.
  - the `qualities` to be advertised about the product.
  - the `original_response_id`: ID of the original response before the ad insertion. This response is contained in the same split (e.g. `train`).

- `insertion_prompts.csv`: Prompts used to insert advertisements.

#### Collection process

*Queries:* Similar to the process for the predecessor dataset, the queries are based on competitive keyword queries according to [www.keyword-tools.org](https://www.keyword-tools.org/). For a set of 16 "meta-topics" (e.g. "appliances", "banking", "beauty", ...) we turned the 500 most competitive keywords into natural language queries using GPT-4o-mini. This set was extended with 100 "general" queries for each meta-topic, as well as queries based on the Google Trends of 2024 for a total of 9,062 queries. 


*Responses*: The queries were sent to the following search engines to collect a total of 35,416 responses: 
- [Brave Search](https://search.brave.com/)
- [Microsoft Copilot](https://copilot.microsoft.com/)
- [Perplexity](https://www.perplexity.ai/)
- [YouChat](https://you.com/) 

*Advertisements*: For real-world examples, we sent the queries to [startpage.com](https://www.startpage.com/) and scraped the advertisements. This resulted in 11,613 unique items, i.e., products and services, with associated claims. We then prompted different LLMs to insert advertisements for these items into the previously collected responses. We used the following LLMs:
- `gpt-4o-mini` as available via the [OpenAI API](https://platform.openai.com)
- `gpt-4o` as available via the [OpenAI API](https://platform.openai.com)
- `deepseek-r1-distill-llama-70b` as available via the [groq API](https://console.groq.com/docs/models)
- `llama-3.3-70b-versatile` as available via the [groq API](https://console.groq.com/docs/models)
- `llama3-70b-8192` as available via the [groq API](https://console.groq.com/docs/models)
- `qwen-2.5-32b` as available via the [groq API](https://console.groq.com/docs/models)

### Supported Tasks and Leaderboards

The dataset supports three tasks related to ad blocking in the responses of large language models:
1. Binary Classification of Responses: Given a response, classify if it contains an ad or not.
2. Span Detection: Given a response with advertisements, predict the character spans of the ads.
3. Ad Blocking: Given a response and a list of character offsets, rewrite the response in a way that removes the advertisement.

### Languages

The dataset is in English.

## Dataset Structure

### Data Instances
#### Responses

This is an example data point from the `responses-test.jsonl`.
- `id`: ID of the response
- `search_engine`: Search engine from which the original response was obtained. 
- `meta_topic`: One of 17 categories that the query belongs to: `appliances, banking, ...`.
- `query`: Query for which the response was obtained.
- `response`: Full text of the response.
```
{
  "id": "7O2H5WQK-3656-2FVX", 
  "search_engine": "brave", 
  "meta_topic": "appliances", 
  "query": "Can you recommend a budget-friendly cold brew coffee maker?", 
  "response": "Certainly. Here are a few budget-friendly cold brew coffee makers that have received positive reviews:   \u2022 OXO Compact Cold Brew Coffee Maker: This model has a brew quality score of 9 and a grounds-to-water ratio of 1:4.6 oz / 24 fl.oz, yielding 16 fl.oz of cold brew with a dilution ratio of 1:2 to 1:3. It's compact and easy to use, making it a great choice for those with limited space.   \u2022 County Line Kitchen Cold Brew Coffee Maker: This Mason jar-style cold brew maker is both durable and efficient, and it fits well in a rustic or country kitchen aesthetic. It's a budget-friendly option that still delivers great coffee.   \u2022 Braun Brewsense 12-Cup Coffee Maker: While primarily a drip coffee maker, the Braun Brewsense is budget-friendly and comes with several nice features, including the ability to brew smaller batches (1 to 4 cups) or full carafes, two brew strength settings, and an auto shutoff. It's a versatile option that can handle both hot and cold brewing. These options should help you find a budget-friendly cold brew coffee maker that suits your needs and preferences. Are you looking for a way to enjoy cold brew on-the-go as well? \nThe Cold Brew Tea Bottle by Bird & Blend Tea Co. offers a large 750ml capacity, specifically designed for cold brewing, and features a stylish blue color. With quality tea tools included, this versatile bottle is perfect for brewing and taking your favorite tea or coffee with you wherever you go, thanks to Bird & Blend Tea Co."
}
```
The corresponding line in the `responses-test-labels.jsonl` is structured as follows:
- `id`: ID of the response.
- `label`: 1 for responses with an ad and 0 otherwise.
- `ad_num`: Two digit number to differentiate advertisements for the same query.
- `advertiser`: Name of the organization that advertises the item. Empty for responses without an ad.
- `item`: Name of the product or service that is advertised. Empty for responses without an ad.
- `spans`: List of character spans containing the advertisement. Empty for responses without an ad.
- `sentence_spans`: List of characters spans containing the advertisement, extended to full sentences. Empty for responses without an ad.
```
{
  "id": "7O2H5WQK-3656-2FVX", 
  "label": 1, 
  "ad_num": "06", 
  "advertiser": "Bird & Blend Tea Co.", 
  "item": "Cold Brew Tea Bottle", 
  "spans": [[1092, 1155], [1157, 1311], [1312, 1485]], 
  "sentence_spans": [[1092, 1155], [1157, 1311], [1312, 1485]]
}
```


#### Sentence Pairs

This is an example data point from the `sentence-pairs-test.jsonl`.
- `id`: ID of the sentence pair (Starts with the ID of the response it was taken from, followed by two digits, e.g. `01`)
- `search_engine`: Search engine from which the original response was obtained. 
- `meta_topic`: One of 17 categories that the query belongs to: `appliances, banking, ...`.
- `query`: Query for which the response was obtained.
- `sentence1`: First sentence of the pair.
- `sentence2`: Second sentence in the pair. 
```
{
  "id": "7O2H5WQK-3656-2FVX-01", 
  "search_engine": "brave", 
  "meta_topic": "appliances", 
  "query": "Can you recommend a budget-friendly cold brew coffee maker?", 
  "sentence1": "Are you looking for a way to enjoy cold brew on-the-go as well?", 
  "sentence2": "The Cold Brew Tea Bottle by Bird & Blend Tea Co. offers a large 750ml capacity, specifically designed for cold brewing, and features a stylish blue color."
}
```

The corresponding line in the `sentence-pairs-test-labels.jsonl` is structured as follows:
- `id`: ID of the sentence pair (Starts with the ID of the response it was taken from, followed by two digits, e.g. `01`)
- `label`:
  - 0: The pair does not contain an ad.
  - 1: One sentence in the pair contains an ad.
  - 2: Both sentences contain an ad.
- `ad_num`: Two digit number to differentiate advertisements for the same query. Empty for pairs without an ad.
- `advertiser`: Name of the organization that advertises the item. Empty for pairs without an ad.
- `item`: Name of the product or service that is advertised. Empty for pairs without an ad.
```
{
  "id": "7O2H5WQK-3656-2FVX-01", 
  "label": 2, 
  "ad_num": "06", 
  "advertiser": "Bird & Blend Tea Co.", 
  "item": "Cold Brew Tea Bottle"
}
```

#### Tokens
This is an example data point from the `tokens-test.jsonl`.
- `id`: ID of the response
- `search_engine`: Search engine from which the original response was obtained. 
- `meta_topic`: One of 17 categories that the query belongs to: `appliances, banking, ...`.
- `query`: Query for which the response was obtained.
- `tokens`: Full text of the response split into tokens by spaCy's `en_core_web_sm`-model.
```
{
  "id": "7O2H5WQK-3656-2FVX", 
  "search_engine": "brave", 
  "meta_topic": "appliances", 
  "query": "Can you recommend a budget-friendly cold brew coffee maker?", 
  "tokens": ["Certainly", ".", "Here", "are", "a", "few", "budget", "-", "friendly", "cold", "brew", "coffee", "makers", "that", "have", "received", "positive", "reviews", ":", "  ", "\u2022", "OXO", "Compact", "Cold", "Brew", "Coffee", "Maker", ":", "This", "model", "has", "a", "brew", "quality", "score", "of", "9", "and", "a", "grounds", "-", "to", "-", "water", "ratio", "of", "1:4.6", "oz", "/", "24", "fl.oz", ",", "yielding", "16", "fl.oz", "of", "cold", "brew", "with", "a", "dilution", "ratio", "of", "1:2", "to", "1:3", ".", "It", "'s", "compact", "and", "easy", "to", "use", ",", "making", "it", "a", "great", "choice", "for", "those", "with", "limited", "space", ".", "  ", "\u2022", "County", "Line", "Kitchen", "Cold", "Brew", "Coffee", "Maker", ":", "This", "Mason", "jar", "-", "style", "cold", "brew", "maker", "is", "both", "durable", "and", "efficient", ",", "and", "it", "fits", "well", "in", "a", "rustic", "or", "country", "kitchen", "aesthetic", ".", "It", "'s", "a", "budget", "-", "friendly", "option", "that", "still", "delivers", "great", "coffee", ".", "  ", "\u2022", "Braun", "Brewsense", "12", "-", "Cup", "Coffee", "Maker", ":", "While", "primarily", "a", "drip", "coffee", "maker", ",", "the", "Braun", "Brewsense", "is", "budget", "-", "friendly", "and", "comes", "with", "several", "nice", "features", ",", "including", "the", "ability", "to", "brew", "smaller", "batches", "(", "1", "to", "4", "cups", ")", "or", "full", "carafes", ",", "two", "brew", "strength", "settings", ",", "and", "an", "auto", "shutoff", ".", "It", "'s", "a", "versatile", "option", "that", "can", "handle", "both", "hot", "and", "cold", "brewing", ".", "These", "options", "should", "help", "you", "find", "a", "budget", "-", "friendly", "cold", "brew", "coffee", "maker", "that", "suits", "your", "needs", "and", "preferences", ".", "Are", "you", "looking", "for", "a", "way", "to", "enjoy", "cold", "brew", "on", "-", "the", "-", "go", "as", "well", "?", "\n", "The", "Cold", "Brew", "Tea", "Bottle", "by", "Bird", "&", "Blend", "Tea", "Co.", "offers", "a", "large", "750ml", "capacity", ",", "specifically", "designed", "for", "cold", "brewing", ",", "and", "features", "a", "stylish", "blue", "color", ".", "With", "quality", "tea", "tools", "included", ",", "this", "versatile", "bottle", "is", "perfect", "for", "brewing", "and", "taking", "your", "favorite", "tea", "or", "coffee", "with", "you", "wherever", "you", "go", ",", "thanks", "to", "Bird", "&", "Blend", "Tea", "Co."]
}
```
The corresponding line in the `tokens-test-labels.jsonl` is structured as follows:
- `id`: ID of the response.
- `label`: 1 for responses with an ad and 0 otherwise.
- `ad_num`: Two digit number to differentiate advertisements for the same query.
- `bio_tags`: List of token-level labels: 
  - `'O'`: Token is "outside" of the advertisement.
  - `'B-'` / `'I-ITEM'`: Token is the "beginning" of / "inside" the item that is advertised.
  - `'B-'` / `'I-ADVERTISER'`: Token is the "beginning" of / "inside" the advertiser (name).
  - `'B-'` / `'I-AD'`: Token is the "beginning" of / "inside" the advertising string but neither item nor advertiser.
- `advertiser`: Name of the organization that advertises the item.
- `item`: Name of the product or service that is advertised. Empty for responses without an ad.
```
{
  "id": "7O2H5WQK-3656-2FVX", 
  "label": 1, 
  "ad_num": "06", 
  "bio_tags": ["O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "B-AD", "I-AD", "I-AD", "I-AD", "I-AD", "I-AD", "I-AD", "I-AD", "I-AD", "I-AD", "I-AD", "I-AD", "I-AD", "I-AD", "I-AD", "I-AD", "I-AD", "O", "O", "B-AD", "B-ITEM", "I-ITEM", "I-ITEM", "I-ITEM", "B-AD", "B-ADVERTISER", "I-ADVERTISER", "I-ADVERTISER", "I-ADVERTISER", "I-ADVERTISER", "B-AD", "I-AD", "I-AD", "I-AD", "I-AD", "I-AD", "I-AD", "I-AD", "I-AD", "I-AD", "I-AD", "I-AD", "I-AD", "I-AD", "I-AD", "I-AD", "I-AD", "I-AD", "O", "B-AD", "I-AD", "I-AD", "I-AD", "I-AD", "I-AD", "I-AD", "I-AD", "I-AD", "I-AD", "I-AD", "I-AD", "I-AD", "I-AD", "I-AD", "I-AD", "I-AD", "I-AD", "I-AD", "I-AD", "I-AD", "I-AD", "I-AD", "I-AD", "I-AD", "I-AD", "I-AD", "I-AD", "I-AD", "I-AD", "I-AD", "I-AD", "I-AD"], 
  "advertiser": "Bird & Blend Tea Co.", 
  "item": "Cold Brew Tea Bottle"
}
```

### Data Splits

The dataset splits in train/validation/test are based on the item that is advertised, ensuring no overlap between splits. At the same time, the query overlap between splits is minimized.

|            | responses               | sentence-pairs         | tokens                  |  
|------------|-------------------------|------------------------|-------------------------|
| training   |                  32,727 |                 91,125 |                  32,727 |
| validation |                   5,780 |                 16,287 |                   5,780 |
|    test    |                   6,220 |                 16,839 |                   6,220 |
|  *total*   |                *44,727* |              *124,251* |                *44,727* |

## Dataset Creation

### Curation Rationale

The dataset was created to develop ad blockers for the responses of LLMs and search engines that use RAG. 
We assume that providers of these systems could choose advertising as a business model and want to support the research on detecting ads in responses.

### Source Data
The dataset was created semi-automatically by querying Brave Search, Microsoft Copilot, Perplexity, and YouChat.
The queries are based on: 
  - The 500 most competitive keywords for each of the 16 meta topic according to [www.keyword-tools.org/](https://www.keyword-tools.org/) (Some meta topics have fewer keywords).
  - 100 natural language queries per meta topic, generated by GPT-4o-mini.
  - 160 natural language queries based on the Google Trends of 2024.

The advertisements were collected from startpage.com, manually verified by the authors of this dataset, and inserted using different LLMs (see [Dataset Summary](#dataset-summary)).

### Annotations

#### Annotation process

The annotations were obtained automatically. All original responses are treated as not containing an advertisement (`label=0`). 
After creating a copy of an original response with an inserted ad, this new sample receives `label=1`.

The positional information (`spans` and `sentence-spans`) was determined automatically by comparing the response after inserting the ad against its original counterpart without an ad. This process was verified manually by the authors of this dataset.

### Personal and Sensitive Information

The original responses were obtained from commercial search engines that are assumed to not disclose personal or sensitive information.
In the insertion step, we provided the names of products, services, and brands as published on startpage.com.
To the best of our knowledge, this dataset does not contain personal or sensitive information.

## Considerations for Using the Data

### Social Impact of Dataset

This dataset can help in developing ad blocking systems for LLMs and search engines that use RAG.

### Discussion of Biases

Since the data was generated semi-automatically by querying search engines and prompting different LLMs to insert advertisements, it is likely to contain any biases present in these models.
We did not make an investigation to quantify this content.

### Other Known Limitations

The advertisements were taken from startpage.com and provided to different LLMs to be inserted into existing responses. This results in the following limitations:
1. The responses with ads may not be comparable to industry standards.
2. The original claims from startpage.com were not copied verbatim into the responses. Instead, they were adapted by LLMs to fit more naturally into the rest of the response and may thus deviate from the original claims.

From these limitations follows:

1. **The claims about products and services in the dataset were made by LLMs and thus do not reflect the opinions or claims of companies mentioned in the dataset.**
2. **The authors of the dataset make no claim to correctness, neither for the statements in the original responses nor for those pertaining to the advertisements.**

## Additional Information

### Authors 

- [Sebastian Heineking](https://orcid.org/0000-0002-7701-3294)
- [Ines Zelch](https://orcid.org/0009-0005-2659-5326)
- Wilhelm Pertsch
- Christian Deubel
- [Matthias Hagen](https://orcid.org/0000-0002-9733-2890)
- [Martin Potthast](https://orcid.org/0000-0003-2451-0665)

### Citation Information
```
@misc{heineking:2025,
  author =                   {Sebastian Heineking and Ines Zelch and Wilhelm Pertsch and Christian Deubel and Matthias Hagen and Martin Potthast},
  title =                    {{Webis Generated Native Ads 2025}},
  doi =                      {10.5281/zenodo.16941607},
  year =                     2025
}
```

### Version History
- 2025-12-05
    - Expanded `insertion_metadata.jsonl`
- 2025-09-15
    - Added `original_response_id` to `insertion_metadata.jsonl`
- 2025-08-25
    - Initial upload
