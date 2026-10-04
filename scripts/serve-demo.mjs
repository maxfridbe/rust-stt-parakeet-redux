// Local-only static server with an explicit allowlist of asset directories.
import { createServer } from "node:http";
import { createReadStream } from "node:fs";
import { stat } from "node:fs/promises";
import { resolve, sep, extname } from "node:path";

const port = Number(process.env.PORT ?? 8080);
const roots = {
  "/demo/": resolve("dist/demo"),
  "/web/": resolve("dist/web"),
  "/models/": resolve("models"),
  "/fixtures/": resolve("tests/fixtures"),
};
const types = {
  ".html": "text/html",
  ".css": "text/css",
  ".js": "text/javascript",
  ".json": "application/json",
  ".wasm": "application/wasm",
  ".pcm": "application/octet-stream",
};
createServer(async (request, response) => {
  try {
    const pathname = decodeURIComponent(
      new URL(request.url, "http://localhost").pathname,
    );
    if (pathname === "/") {
      response.writeHead(302, { Location: "/demo/" });
      response.end();
      return;
    }
    const prefix = Object.keys(roots).find((prefix) =>
      pathname.startsWith(prefix),
    );
    if (!prefix) {
      response.writeHead(404);
      response.end("Not found");
      return;
    }
    const root = roots[prefix];
    const path = resolve(root, pathname.slice(prefix.length) || "index.html");
    if (!path.startsWith(root + sep)) {
      response.writeHead(403);
      response.end("Forbidden");
      return;
    }
    const info = await stat(path);
    if (!info.isFile()) {
      response.writeHead(404);
      response.end("Not found");
      return;
    }
    response.writeHead(200, {
      "Content-Type": types[extname(path)] ?? "application/octet-stream",
      "Content-Length": info.size,
      "Cache-Control": "no-cache",
    });
    createReadStream(path).pipe(response);
  } catch {
    response.writeHead(404);
    response.end("Not found");
  }
}).listen(port, "127.0.0.1", () =>
  console.log(`Microphone demo: http://localhost:${port}/demo/`),
);
