# Hackton Group G031 CodeBots
# Topic : GEN-AI 
# Problem Statement: A company struggles to reach and engage its target audience resulting in suboptimal brand visibility and low conversion rates. 
#                    Develop a solution that leverages generativeAI to enhance brand visibility improve audience targeting and increase both engagement and conversion

# Implemented Solution : 
import gradio as gr
from transformers import pipeline
import spacy
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch


generator = pipeline('text-generation', model='gpt2')
nlp = spacy.load("en_core_web_sm")


def generate_related_keywords(seed_keyword):
    prompt = f"Generate related keywords for the search term '{seed_keyword}'."
    result = generator(prompt, max_length=50, num_return_sequences=1)
    keywords = result[0]['generated_text']
    return keywords


def analyze_search_intent(query):
    prompt = f"What is the search intent behind the query: '{query}'?"
    result = generator(prompt, max_length=50, num_return_sequences=1)
    return result[0]['generated_text']


def optimize_content(content):
    prompt = f"Improve the SEO of the following content, making it more engaging and keyword-rich:\n{content}"
    result = generator(prompt, max_length=200, num_return_sequences=1)
    return result[0]['generated_text']


def generate_meta_tags(content):
    prompt_title = f"Create an SEO-friendly title for the following content: {content}"
    prompt_description = f"Create an SEO-friendly meta description for the following content: {content}"
    
    title = generator(prompt_title, max_length=60, num_return_sequences=1)[0]['generated_text']
    description = generator(prompt_description, max_length=160, num_return_sequences=1)[0]['generated_text']
    
    return title, description


def improve_readability(text):
    doc = nlp(text)
    readability = [sent.text for sent in doc.sents if len(sent.text.split()) < 20]
    return readability


def audit_page_seo(content):
    prompt = f"Analyze this content for SEO. Suggest improvements for keyword usage, readability, and metadata: {content}"
    result = generator(prompt, max_length=200, num_return_sequences=1)
    return result[0]['generated_text']


def ner_feedback(content):
    doc = nlp(content)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    feedback = "Identify how to leverage entities in the content to improve conversion:\n"
    
    for entity, label in entities:
        if label == "ORG":
            feedback += f"Consider emphasizing the organization {entity} to improve brand visibility.\n"
        elif label == "PRODUCT":
            feedback += f"Focus on {entity} as a key selling point to engage customers.\n"
        elif label == "GPE":
            feedback += f"Tailor the content to the audience in {entity} for better regional targeting.\n"
    
    return feedback


def seo_optimizer(seed_keyword, content):

    keywords = generate_related_keywords(seed_keyword)
    
   
    intent = analyze_search_intent(seed_keyword)
    
    optimized_content = optimize_content(content)
    
    meta_title, meta_description = generate_meta_tags(content)
    
    readable_content = improve_readability(content)
    
    audit_report = audit_page_seo(content)
    
   
    conversion_feedback = ner_feedback(content)
    
    return (keywords, intent, optimized_content, meta_title, meta_description, 
            "\n".join(readable_content), audit_report, conversion_feedback)


model_id = "stabilityai/stable-diffusion-2"
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    variant="fp16", 
    scheduler=scheduler,
    torch_dtype=torch.float16,
    safety_checker=None,  
    allow_pickle=True  
)
pipe = pipe.to("cuda") 


def txt2img(prompt):
    image = pipe(prompt, height=768, width=768, guidance_scale=10).images[0]
    image.save("sd_image.png")
    return image


def recommend_based_on_preferences(seed_keyword, content, user_preferences, image_prompt):
   
    if 'short' in user_preferences:
        content = content[:200]  
    if 'detailed' in user_preferences:
        content = content + " This is a detailed extension of the content."

    seo_results = seo_optimizer(seed_keyword, content)

 
    generated_image = txt2img(image_prompt)
    
    return (*seo_results, generated_image)


interface = gr.Interface(
    fn=recommend_based_on_preferences,
    inputs=[
        gr.Textbox(label="Seed Keyword", placeholder="Enter seed keyword", lines=1),
        gr.Textbox(label="Content", placeholder="Enter content to optimize", lines=5),
        gr.CheckboxGroup(label="User Preferences", choices=["short", "detailed", "SEO-focused", "engaging", "keyword-rich"]),
        gr.Textbox(label="Image Prompt", placeholder="Enter prompt for image generation", lines=2),
    ],
    outputs=[
        gr.Textbox(label="Related Keywords"),
        gr.Textbox(label="Search Intent"),
        gr.Textbox(label="Optimized Content"),
        gr.Textbox(label="Meta Title"),
        gr.Textbox(label="Meta Description"),
        gr.Textbox(label="Readable Content"),
        gr.Textbox(label="SEO Audit Report"),
        gr.Textbox(label="Customer Conversion Feedback"),  
        gr.Image(label="Generated Image")
    ],
    live=True 
)

interface.launch(debug=True, share=True)
