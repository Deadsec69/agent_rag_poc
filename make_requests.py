import asyncio
import aiohttp
from typing import List, Optional
from aiohttp import FormData


BASE_URL = "http://localhost:8000"  # Adjust this if your server is running on a different address

async def query(question: str) -> str:
    """
    Send a query to the /query endpoint and stream the response.
    """
    url = f"{BASE_URL}/query"
    data = {"query": question}
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=data) as response:
            if response.status == 200:
                full_response = ""
                async for chunk in response.content.iter_any():
                    chunk_str = chunk.decode('utf-8')
                    print(chunk_str, end='', flush=True)
                    full_response += chunk_str
                print()  # New line after the full response
                return full_response
            else:
                print(f"Error: {response.status}")
                return await response.text()

async def train(urls: Optional[List[str]] = None, 
                pdf_files: Optional[List[str]] = None, 
                text_files: Optional[List[str]] = None) -> str:
    """
    Send a request to the /train endpoint with the provided data.
    """
    url = f"{BASE_URL}/train"
    
    form = FormData()
    
    if urls:
        form.add_field('urls', ','.join(urls))
    
    if pdf_files:
        for pdf_file in pdf_files:
            form.add_field('pdf_files', 
                           open(pdf_file, 'rb'),
                           filename=pdf_file,
                           content_type='application/pdf')
    
    if text_files:
        for text_file in text_files:
            form.add_field('text_files', 
                           open(text_file, 'rb'),
                           filename=text_file,
                           content_type='text/plain')
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, data=form) as response:
            if response.status == 200:
                return await response.json()
            else:
                print(f"Error: {response.status}")
                return await response.text()

async def main():
    # Example usage
    question = "What is the capital of France?"
    print(f"Querying: {question}")
    await query(question)
    
    print("\nTraining with example data:")
    result = await train(
        urls=["https://radissonhotels.iceportal.com/asset/radisson-blu-hotel-ranchi/miscellaneous/16256-114065-m28695768.pdf"],
        pdf_files=["pdfs/Hotel-Brochure-Vivanta-New-Delhi-Dwarka.pdf"],
        text_files=["txts/oberoi.txt"]
    )
    print(result)
    question = "Write a small blog about Vivanta"
    print(f"Querying: {question}")
    await query(question)

if __name__ == "__main__":
    asyncio.run(main())