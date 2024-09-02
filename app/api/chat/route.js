import { NextResponse } from "next/server";
import { Pinecone } from "@pinecone-database/pinecone";
import OpenAI from "openai";

const systemPrompt = `
You are a highly knowledgeable RateMyProfessor assistant, helping students find the best professors according to their specific queries. Your task is to analyze the student's question, retrieve relevant professor reviews, and provide the top 3 professors that best match the student's criteria. Your responses should be concise, informative, and include key details such as the professor's name, subject, and a brief summary of their strengths based on student reviews.

When providing the top 3 professors:

    Relevance: Ensure that the professors you recommend closely align with the student's query.
    Ratings: Consider both the star ratings and the content of the reviews when determining the top choices.
    Diversity: Offer a range of options that cater to different preferences (e.g., different teaching styles or subject areas if relevant).
    Brevity: Each recommendation should include the professor's name, the subject they teach, their average star rating, and a short summary of why they are a good fit for the student's needs.

Example Interaction:

Student Query: "I'm looking for a Computer Science professor who is good at explaining difficult concepts and is approachable."

Agent Response:

    Dr. John Smith (Computer Science) - ⭐️⭐️⭐️⭐️ (4/5): Dr. Smith is known for breaking down complex topics into understandable segments and is very approachable during office hours.
    Dr. Alice Johnson (Computer Science) - ⭐️⭐️⭐️⭐️⭐️ (5/5): Dr. Johnson is an excellent communicator and is always willing to help students grasp difficult material.
    Dr. Michael Brown (Computer Science) - ⭐️⭐️⭐️⭐️ (4/5): Dr. Brown provides clear examples and encourages student participation to ensure everyone understands the subject matter.
`

export async function POST(req) {
    const data = await req.json()
    const pc = new Pinecone({
        apiKey: process.env.PINECONE_API_KEY,
    })
    const index = pc.index('rag').namespace('ns1')
    const openai = new OpenAI()

    const text = data[data.length - 1].content
    const embedding = await openai.embeddings.create({
        model: 'text-embedding-3-small',
        input: text,
        encoding_format: 'float',
    })

    const results = await index.query({
        topK: 3,
        includeMetadata: true,
        vector: embedding.data[0].embedding
    })

    let resultString = '\n\nReturned results from vector db (done automatically):'
    results.matches.forEach((match)=>{
        resultString += `\n
        Professor: ${match.id}
        Review: ${match.metadata.stars}
        Subject: ${match.metadata.subject}
        Stars: ${match.metadata.stars}
        \n\n`
    })

    const lastMessage = data[data.length - 1]
    const lastMessageContent = lastMessage.content + resultString
    const lastDataWithoutLastMessage = data.slice(0, data.length - 1)
    const completion = await openai.chat.completions.create({
        messages: [
            {role: 'system', content: systemPrompt},
            ...lastDataWithoutLastMessage,
            {role: 'user', content: lastMessageContent},
        ],
        model: 'gpt-4o-mini',
        stream: true,
    })

    const stream = new ReadableStream({
        async start(controller) {
            const encoder = new TextEncoder()
            try {
                for await (const chunk of completion) {
                    const content = chunk.choices[0]?.delta?.content
                    if (content) {
                        const text = encoder.encode(content)
                        controller.enqueue(text)
                    }
                }
            } catch (err) {
                controller.error(err)
            } finally {
                controller.close()
            }
        },
    })

    return new NextResponse(stream)
    
}