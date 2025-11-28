import asyncio
from fastmcp import Client

client = Client("http://localhost:8000/mcp")

async def call_tool(prompt: str):
    async with client:
        result = await client.call_tool("generate_image", {"prompt": prompt})
        print(result)

asyncio.run(call_tool("A futuristic cityscape at sunset"))