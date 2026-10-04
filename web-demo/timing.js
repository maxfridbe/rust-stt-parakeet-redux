// Window and Worker have different time origins. Use comparable monotonic times.
export const timestamp = () => performance.timeOrigin + performance.now();

export function pipelineTiming(job, result, receivedAt) {
  const elapsed = (end, start) => Math.max(0, end - start);
  return {
    collectionMs:
      job.audioStartedAt == null
        ? null
        : elapsed(job.enqueuedAt, job.audioStartedAt),
    speechWaitMs:
      job.speechEndedAt == null
        ? null
        : elapsed(job.enqueuedAt, job.speechEndedAt),
    queueMs: elapsed(job.processingStartedAt, job.enqueuedAt),
    preparationMs: elapsed(job.dispatchedAt, job.processingStartedAt),
    dispatchMs: elapsed(result.startedAt, job.dispatchedAt),
    inferenceMs: result.inferenceMs,
    deliveryMs: elapsed(receivedAt, result.finishedAt),
    submittedToResultMs: elapsed(receivedAt, job.enqueuedAt),
    speechToResultMs:
      job.speechEndedAt == null ? null : elapsed(receivedAt, job.speechEndedAt),
  };
}
