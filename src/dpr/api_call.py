import os

from openai import OpenAI

OPENAI_API_KEY = "your_openai_api_key"


def gpt_api(messages, model_name="gpt-4o-mini"):
    client = OpenAI()

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0
    )

    return response.choices[0].message.content


# prompts for keyword generation
def get_general_keywords(query, num=2):
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    system_prompt = f"""# Instruction: Generate a concise set of precise and domain-specific keywords up to {num} based on the given query.
Focus only on the most critical and high-impact keywords that are essential for understanding or addressing the query.
Each keyword should represent the core elements of the query, such as key mechanisms, biological processes, or specific interactions.
Exclude minor or overly generic terms like 'relationship' or 'mechanism' unless they are explicitly critical to the query.
Prioritize terms that are unique, actionable, and directly relevant to the domain of the query.
Provide the results as a numbered list in the format: <number>. <keyword>."""
    user_prompt = f"""# Query: {query}
# Output:"""
    
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    response = gpt_api(messages)
    if "# Output:" in response:
        response = response.replace("# Output:", " ")
    return response.lower().strip()


def get_mesh_keywords(query, num=2):
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    system_prompt = f"""# Instruction: Generate a set of keywords up to {num} based on the given query, leveraging the MeSH (Medical Subject Headings) vocabulary. The keywords should be derived from relevant MeSH terms or closely related concepts.
Reflect the core elements of the query, including biological processes, diseases, genes, proteins, and pathways.
Avoid overly generic terms unless explicitly relevant to the query.
Prioritize specific MeSH terms that align directly with the query and terms that are actionable and provide deeper insight into the topic.
Output the results as a numbered list in the format <number>. <MeSH term>."""
    user_prompt = f"""# Query: {query}
# Output:"""
    
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    response = gpt_api(messages)
    if "# Output:" in response:
        response = response.replace("# Output:", " ")
    return response.lower().strip()


# prompts for query transformation
def get_virtual_abstract(query):
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    system_prompt = f"""# Instruction: Generate only one abstract of a biology or medical paper that can solve the given query.
Ensure that the abstract is logically structured, concise, and relevant to the query. Provide the abstract text as plain text without titles, numbering, or additional formatting."""
    user_prompt = f"""# Query: {query}
# Output:"""
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    response = gpt_api(messages)
    if "# Output:" in response:
        response = response.replace("# Output:", " ")
    return response.lower().strip()


def get_rewrited_query(query, keyword=None):
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    system_prompt = f"""# Instruction: Generate a refined and answerable question based on the given query. The revised question should eliminate ambiguity, focus on concrete and practical aspects, and ensure clarity in scope and intent. Avoid abstract or overly philosophical interpretations, and instead reframe the query to facilitate a clear and actionable response."""
    user_prompt = f"""# Query: {query}
# Output:"""
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    response = gpt_api(messages)
    if "# Output:" in response:
        response = response.replace("# Output:", " ")
    return response.lower().strip()


def get_abstract_from_keyword(query, keyword):
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    system_prompt = f"""# Instruction: Generate only one abstract of a biology or medical paper that can solve the given query based on the given keyword. The abstract should directly address the query while focusing solely on the perspective of the keyword. Ensure that the abstract is logically structured, concise, and relevant to both the query and aspect. Provide the abstract text as plain text without titles, numbering, or additional formatting."""
    user_prompt = f"""# Query: {query}
# Keyword: {keyword}
# Output:"""
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    response = gpt_api(messages)
    if "# Output:" in response:
        response = response.replace("# Output:", " ")
    return response.lower()


def get_query_from_keyword(query, keyword):
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    system_prompt = f"""# Instruction: Generate only one question based on the given query. The question should focus on the given keyword of the topic. Output only the question text as plain text, without any titles, numbering, or additional formatting."""
    user_prompt = f"""# Query: {query}
# Keyword: {keyword}
# Output:"""
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    response = gpt_api(messages)
    if "# Output:" in response:
        response = response.replace("# Output:", " ")
    return response.lower().strip()


# prompts for evaluation
def eval_factuality(query, response):
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    
    system_prompt = f"""# Instruction: Evaluate the generated response based on Factuality, which measures the extent to which the response provides accurate, truthful, and verifiable information without introducing false, misleading, or hallucinated content in relation to the given query.
Use the following 5-point scale to assign a Factuality score (1 to 5 points):

### 1 Point (Very Low Factuality)
- The response contains multiple factual errors, hallucinations, or clearly incorrect statements.
- It misrepresents key facts and provides false or misleading information that may significantly harm understanding or decision-making.

### 2 Points (Low Factuality)
- The response has some accurate content, but also includes important factual inaccuracies, unsupported claims, or misleading generalizations.
- These issues reduce the reliability of the response.

### 3 Points (Moderate Factuality)
- The response is mostly accurate, but may contain minor factual inaccuracies, vague statements, or claims lacking clear evidence.
- While not seriously misleading, it requires some fact-checking or clarification.

### 4 Points (High Factuality)
- The response is factually accurate overall, with reliable and verifiable content.
- There may be minor omissions or simplifications, but no false or misleading claims.

### 5 Points (Very High Factuality)
- The response is entirely accurate, well-supported by scientific evidence, and free from errors, hallucinations, or exaggeration.
- All claims are trustworthy and align with established biomedical knowledge.

## Evaluation Task:
After reviewing the response, assign a Factuality score (1 to 5 points) based on the scale above.
Only provide the score. No justification is required."""
    user_prompt = f"""# Query: {query}
# Response: {response}
# Output:"""

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    response = gpt_api(messages, model_name="gpt-4o")
    if "# Output:" in response:
        response = response.replace("# Output:", "")
    return response.strip()


def eval_diversity(query, response):
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    system_prompt = f"""# Instruction: Evaluate the generated response based on Diversity, which measures the extent to which the response integrates a broad range of relevant information, perspectives, biological mechanisms, diagnostic approaches, methodologies, or treatment strategies to effectively address the given query.
Use the following 5-point scale to assign a Diversity score (1 to 5):

### 1 Point (Very Low Diversity)
- The response relies heavily on a single concept, source, or perspective, with little to no exploration of other relevant aspects or alternative viewpoints.
- Critical dimensions relevant to the query, such as underlying mechanisms, diagnostic considerations, or therapeutic options, are largely absent.

### 2 Points (Low Diversity)
- The response mentions more than one piece of information, but coverage is narrow, repetitive, or focused on a single aspect.
- Other important aspects (e.g., additional biological pathways, diagnostic tools, or treatment approaches) are underdeveloped or missing.

### 3 Points (Moderate Diversity)
- The response addresses multiple relevant aspects related to the query, such as mechanisms, diagnostic approaches, or treatment strategies, but lacks depth or balance in some areas.
- There is some variety, but the range of perspectives or approaches could be expanded to provide a more comprehensive understanding.

### 4 Points (High Diversity)
- The response incorporates a variety of relevant biological, diagnostic, and/or therapeutic perspectives, covering most major aspects related to the query.
- The response is well-rounded, with only minor gaps or slight imbalances in coverage.

### 5 Points (Very High Diversity)
- The response synthesizes a comprehensive range of biological, diagnostic, and therapeutic aspects, drawing from multiple perspectives and approaches.
- All critical dimensions relevant to the query are addressed, and the information is balanced and mutually complementary.

## Evaluation Task:
After reviewing the response, assign a Diversity score (1 to 5 points) based on the scale above.
Only provide the score. No justification is required."""
    user_prompt = f"""# Query: {query}
# Response: {response}
# Output:"""
    
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    response = gpt_api(messages, model_name="gpt-4o")
    if "# Output:" in response:
        response = response.replace("# Output:", "")
    return response.strip()


def eval_clarity(query, response):
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    system_prompt = f"""# Instruction: Evaluate the generated response based on Clarity, which measures the extent to which the response is expressed in a clear, well-structured, and easy-to-understand manner, helping the reader grasp the solution to the given query without confusion.
Use the following 5-point scale to assign a Clarity score (1 to 5):

### 1 Point (Very Low Clarity)
- The response is highly difficult to understand, disorganized, or contains excessive jargon and unclear explanations.  
- The core message is obscured, making it hard to grasp how it addresses the query.

### 2 Points (Low Clarity)
- The response is partially understandable, but contains confusing sections, poor structure, or vague language.  
- Significant effort is required to comprehend how the response relates to the query.

### 3 Points (Moderate Clarity)
- The response is generally understandable, but some sentences may be difficult to follow, or the structure could be improved for easier reading.  
- Occasional ambiguity or complex phrasing may slightly hinder understanding.

### 4 Points (High Clarity)
- The response is mostly clear and easy to understand, with logical structure and minimal ambiguity.  
- Minor refinements could further enhance readability, but comprehension is smooth overall.

### 5 Points (Very High Clarity)
- The response is exceptionally clear, concise, and well-structured, with straightforward language and smooth logical flow.  
- The reader can easily understand how the response addresses the query without any confusion.

## Evaluation Task:
After reviewing the response, assign a Clarity score (1 to 5 points) based on the scale above.
Only provide the score. No justification is required."""
    
    user_prompt = f"""# Query: {query}
# Response: {response}
# Output:"""
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    response = gpt_api(messages, model_name="gpt-4o")
    if "# Output:" in response:
        response = response.replace("# Output:", "")
    return response.strip()


def eval_insightfulness(query, response):
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    system_prompt = f"""# Instruction: Evaluate the generated response based on Insightfulness, which measures the extent to which the response offers meaningful, valuable, and non-obvious insights that contribute to understanding or addressing the given query.
Use the following 5-point scale to assign an Insightfulness score (1 to 5 points):

### 1 Point (Very Low Insightfulness)
- The response provides little to no new understanding, stating only obvious or generic information.  
- It fails to offer any useful guidance in addressing the query.

### 2 Points (Low Insightfulness)
- The response contains some relevant information, but the insights are shallow or largely restate well-known facts.  
- There is minimal value added to deepen understanding of the query.

### 3 Points (Moderate Insightfulness)
- The response offers a few helpful insights, but they may be somewhat basic, incomplete, or lacking originality.  
- Some value is added, but the response could be more thought-provoking or detailed.

### 4 Points (High Insightfulness)
- The response presents several meaningful insights, going beyond surface-level information.  
- It enhances understanding of the query, though a few aspects might still benefit from deeper exploration.

### 5 Points (Very High Insightfulness)
- The response delivers deep, valuable, and non-obvious insights, offering novel perspectives or synthesizing information in a way that greatly enhances understanding of the query.  
- The response stands out as particularly insightful and well thought out.

## Evaluation Task:
After reviewing the response, assign an Insightfulness score (1 to 5 points) based on the scale above.  
Only provide the score. No justification is required."""
    user_prompt = f"""# Query: {query}
# Response: {response}
# Output:"""
    
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    response = gpt_api(messages, model_name="gpt-4o")
    if "# Output:" in response:
        response = response.replace("# Output:", "")
    return response.strip()


# prompt for response generation
def generate(query, abstracts, keywords=None):
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    if keywords == None:
        system_prompt = """Using the provided abstracts, generate a response that either addresses the given query directly or provides meaningful insights that contribute to understanding or resolving the query."""
        user_prompt = f"""# Query: {query}
# Abstracts: {abstracts}
# Output:"""
    else:
        system_prompt = "Each keyword corresponds to abstracts in one-to-one relationship. Using the provided keywords and their corresponding abstracts, generate a response that either addresses the given query directly or offers meaningful insights that contribute to understanding or resolving the query."

        # prepare keyword string
        keyword_str = ""
        for i, keyword in enumerate(keywords):
            keyword_str += f"{i+1}. {keyword}\n"
        
        # prepare abstract string
        abstract_str = ""
        for i, abstract in enumerate(abstracts):
            abstract_str += f"{i+1}. {abstract}\n"
        
        user_prompt = f"""# Query: {query}
# Keywords: {keyword_str}
# Abstracts: {abstract_str}
# Output:"""

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]
    
    response = gpt_api(messages, model_name="gpt-4o-mini")
    if "# Output:" in response:
        response = response.replace("# Output:", "")
    return response.strip()