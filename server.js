import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import Perplexity, { APIError, AuthenticationError, PermissionDeniedError, RateLimitError } from '@perplexity-ai/perplexity_ai';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PORT = process.env.PORT || 3000;
const MAX_QUESTION_LENGTH = 2000;

if (!process.env.PERPLEXITY_API_KEY) {
  console.error(
    'PERPLEXITY_API_KEY is not set. /api/research will return errors until it is exported in this process\'s environment.'
  );
}

// Constructing without an apiKey throws immediately, so only build the client when the key is present.
const perplexity = process.env.PERPLEXITY_API_KEY ? new Perplexity() : null;

function sendJSON(res, status, body) {
  const data = JSON.stringify(body);
  res.writeHead(status, { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(data) });
  res.end(data);
}

async function readRequestBody(req) {
  const chunks = [];
  let size = 0;
  for await (const chunk of req) {
    size += chunk.length;
    if (size > 32 * 1024) throw Object.assign(new Error('Request body too large'), { statusCode: 413 });
    chunks.push(chunk);
  }
  return Buffer.concat(chunks).toString('utf8');
}

/**
 * Pulls the answer text and source citations out of a Perplexity Agent API
 * response. Reads response.output_text for the answer (the SDK's convenience
 * getter), and walks response.output for citations: url_citation annotations
 * on message content parts, plus any search_results tool output.
 */
function extractAnswer(response) {
  const citations = new Map();

  for (const item of response.output ?? []) {
    if (item.type === 'message') {
      for (const part of item.content ?? []) {
        if (part.type === 'output_text') {
          for (const annotation of part.annotations ?? []) {
            if (annotation.url) citations.set(annotation.url, annotation.title || annotation.url);
          }
        }
      }
    } else if (item.type === 'search_results') {
      for (const result of item.results ?? []) {
        if (result.url) citations.set(result.url, result.title || result.url);
      }
    }
  }

  return {
    id: response.id,
    status: response.status,
    answer: response.output_text ?? '',
    citations: [...citations].map(([url, title]) => ({ url, title })),
  };
}

async function handleResearch(req, res) {
  if (!perplexity) {
    return sendJSON(res, 500, {
      error: 'server_misconfigured',
      message: 'PERPLEXITY_API_KEY is not set on the server. Export it in the server\'s environment and restart.',
    });
  }

  let payload;
  try {
    const raw = await readRequestBody(req);
    payload = JSON.parse(raw);
  } catch (err) {
    return sendJSON(res, err.statusCode === 413 ? 413 : 400, {
      error: 'bad_request',
      message: err.statusCode === 413 ? 'Request body too large.' : 'Body must be valid JSON.',
    });
  }

  const question = typeof payload.question === 'string' ? payload.question.trim() : '';
  if (!question) return sendJSON(res, 400, { error: 'bad_request', message: '"question" is required.' });
  if (question.length > MAX_QUESTION_LENGTH) {
    return sendJSON(res, 400, { error: 'bad_request', message: `"question" must be ${MAX_QUESTION_LENGTH} characters or fewer.` });
  }

  const goalContext = Array.isArray(payload.goalContext)
    ? payload.goalContext.filter((g) => typeof g === 'string').slice(0, 10)
    : [];
  const previousResponseId = typeof payload.previousResponseId === 'string' ? payload.previousResponseId : undefined;

  try {
    const response = await perplexity.responses.create({
      preset: 'low',
      tools: [{ type: 'web_search' }],
      instructions:
        'You help someone research peptide protocols they are considering, alongside a licensed provider. ' +
        'Ground every claim in current, cited web sources. Be direct about where evidence is thin. ' +
        'This is not medical advice and you should say so if the question calls for a clinical judgment.' +
        (goalContext.length ? ` The user's stated goal areas: ${goalContext.join(', ')}.` : ''),
      input: question,
      ...(previousResponseId ? { previous_response_id: previousResponseId } : {}),
    });

    return sendJSON(res, 200, extractAnswer(response));
  } catch (err) {
    // Note: this SDK's generated error classes never override `.name` (it's always
    // the literal string "Error"), so the class itself — not err.name — is what
    // identifies the failure. Use err.constructor.name for logging.
    if (err instanceof AuthenticationError || err instanceof PermissionDeniedError) {
      console.error(`Perplexity API rejected the request (${err.status} ${err.constructor.name}) — check PERPLEXITY_API_KEY.`);
      return sendJSON(res, 500, { error: 'upstream_auth_failed', message: 'Server-side Perplexity authentication failed.' });
    }
    if (err instanceof RateLimitError) {
      const retryAfter = err.headers?.['retry-after'];
      if (retryAfter) res.setHeader('Retry-After', retryAfter);
      return sendJSON(res, 429, {
        error: 'rate_limited',
        message: 'Perplexity API rate limit hit. Please retry shortly.',
        retryAfter: retryAfter ?? null,
      });
    }
    if (err instanceof APIError) {
      console.error('Perplexity API error:', err.status, err.constructor.name);
      return sendJSON(res, 502, { error: 'upstream_error', message: `Perplexity API error (${err.constructor.name}).` });
    }
    console.error('Unexpected error calling Perplexity API:', err);
    return sendJSON(res, 500, { error: 'internal_error', message: 'Unexpected server error.' });
  }
}

async function handleStatic(req, res) {
  if (req.method !== 'GET' || req.url !== '/' && req.url !== '/index.html') {
    return sendJSON(res, 404, { error: 'not_found', message: 'Not found.' });
  }
  const html = await readFile(path.join(__dirname, 'index.html'));
  res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
  res.end(html);
}

const server = createServer((req, res) => {
  if (req.method === 'POST' && req.url === '/api/research') {
    handleResearch(req, res).catch((err) => {
      console.error('Unhandled error in /api/research:', err);
      sendJSON(res, 500, { error: 'internal_error', message: 'Unexpected server error.' });
    });
    return;
  }
  handleStatic(req, res).catch((err) => {
    console.error('Unhandled error serving static file:', err);
    sendJSON(res, 500, { error: 'internal_error', message: 'Unexpected server error.' });
  });
});

server.listen(PORT, () => {
  console.log(`Protocol server running at http://localhost:${PORT}`);
});
