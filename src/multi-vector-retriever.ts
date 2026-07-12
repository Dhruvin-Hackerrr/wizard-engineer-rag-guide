import { Document } from '@langchain/core/documents'
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { CohereEmbeddings } from "@langchain/cohere";
import { PineconeStore } from '@langchain/pinecone';
import { Pinecone as PineconeClient } from '@pinecone-database/pinecone'

import { CheerioWebBaseLoader } from '@langchain/community/document_loaders/web/cheerio';
import * as dotenv from 'dotenv';
import { v4 as uuidv4 } from 'uuid';

dotenv.config()

/**
 * 
 * urls is an @Array
 * docId is a @String
 */
async function loadRawDocs(urls : string[], docId : string){
    const allDocs = await Promise.all(
        urls.map(async (url) => {
            const loader = new CheerioWebBaseLoader(url)
            const docs = await loader.load()

            return docs.map((doc) => {
                doc.metadata.originalUrl = url
                doc.metadata.source = url
                doc.metadata.id = docId
                return doc
            })
        })
    )

    return allDocs.flat()
}

async function createParentDocs(rowDocs : Document[], docId : string){
    const parentSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: 2000,
        chunkOverlap: 400,
    })
    const parentSplits = await parentSplitter.splitDocuments(rowDocs)
    return parentSplits.map((doc)=>{
        const chunkId = uuidv4();
        doc.metadata.docType = "parent";
        doc.metadata.chunkId = chunkId;
        doc.metadata.parentId = chunkId;
        doc.metadata.source = chunkId;
        doc.metadata.id = docId
        return doc
    })

}

async function createChildDocs(parentDocs : Document[], docId : string){
    const childSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: 400,
        chunkOverlap: 50,
    })
    const childSplits = await childSplitter.splitDocuments(parentDocs)

    return childSplits.map((doc, i) => {
        // Get Parent metadata for this child 
        const parentIndex = Math.floor(i / 4)
        const parentMetadata = parentDocs[parentIndex]?.metadata

        doc.metadata.docType = "child";
        doc.metadata.parentId = parentMetadata?.chunkId;

        doc.metadata.chunkId = `child-${parentMetadata?.chunkId}-${i}`;
        doc.metadata.source = doc.metadata.chunkId;
        doc.metadata.id = docId

        return doc
    })
    
}


export async function docEmbeddingMultiVector(urls : string[], docId : string){
    const embeddings = new CohereEmbeddings({
        model: "embed-english-v3.0",
        apiKey: process.env.COHERE_API_KEY as string
    })

    const pinecone = new PineconeClient({
        apiKey: process.env.PINECONE_API_KEY as string,
    })
    const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX as string)

    console.log("Loading raw documents....")
    const rawDocs = await loadRawDocs(urls, docId)
    
    console.log("Creating Parent Documents....")
    const parentDocs = await createParentDocs(rawDocs, docId)
    console.log("parentDoc :", parentDocs)

    console.log("Creating Child Documents....")
    const childDocs = await createChildDocs(parentDocs, docId)
    console.log("childDoc :", childDocs)

    console.log("Storing in Pinecone....")
    const vectorStore = new PineconeStore(embeddings, { pineconeIndex, maxConcurrency: 5 })
    await vectorStore.addDocuments([...parentDocs, ...childDocs])

    console.log(`Single Index : ${parentDocs.length} parent chunks + ${childDocs.length} child chunks`)
    console.log(`Total documents : ${parentDocs.length + childDocs.length}`)
        
}

export async function queryMultiVector(query : string, docId : string){

    const kParents = 3
    const embeddings = new CohereEmbeddings({
        model: "embed-english-v3.0",
        apiKey: process.env.COHERE_API_KEY as string
    })

    const pinecone = new PineconeClient({
        apiKey: process.env.PINECONE_API_KEY as string,
    })
    const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX as string)
    
    const vectorStore = new PineconeStore(embeddings, { pineconeIndex, maxConcurrency: 5 })

    const childDocs = await vectorStore.similaritySearch(query, 10,
        { docType : "child", id : docId}
    )

    const parentChunkIds = [...new Set(childDocs.map((doc) => doc.metadata.parentId))]
    const filteredChunkIds = parentChunkIds.filter((id : string) => id !== undefined && id !== null)

    const parentDocs = await vectorStore.similaritySearch(query, kParents,
        {
            docType : "parent",
            source : { $in : filteredChunkIds }
        }
    )

    return parentDocs
}

// await docEmbeddingMultiVector(["https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/"], "doc1")

const formatDocumentAsString = (documents : Document[]) => {
    return documents.map((doc) => doc?.pageContent).join("\n\n")
}

const result = await queryMultiVector("What is prompt engineering?", "doc1")

const docToString = formatDocumentAsString(result)

console.log(docToString)