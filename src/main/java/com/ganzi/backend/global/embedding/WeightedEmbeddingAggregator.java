package com.ganzi.backend.global.embedding;

import java.util.Optional;
import lombok.extern.slf4j.Slf4j;

@Slf4j
public class WeightedEmbeddingAggregator {

    private final EmbeddingJsonConverter embeddingJsonConverter;
    private float[] sum;
    private double totalWeight;

    public WeightedEmbeddingAggregator(EmbeddingJsonConverter embeddingJsonConverter) {
        this.embeddingJsonConverter = embeddingJsonConverter;
    }

    public void add(String embeddingJson, double weight, String context) {
        embeddingJsonConverter.toVector(embeddingJson, context)
                .ifPresent(vector -> accumulate(vector, weight, context));
    }

    public Optional<float[]> normalizedAverage() {
        if (sum == null || totalWeight == 0.0) {
            return Optional.empty();
        }

        float[] average = new float[sum.length];
        for (int i = 0; i < sum.length; i++) {
            average[i] = sum[i] / (float) totalWeight;
        }

        double norm = 0.0;
        for (float v : average) {
            norm += v * v;
        }

        norm = Math.sqrt(norm);
        if (norm > 0) {
            for (int i = 0; i < average.length; i++) {
                average[i] /= (float) norm;
            }
        }

        return Optional.of(average);
    }

    private void accumulate(float[] vector, double weight, String context) {
        if (sum == null) {
            sum = new float[vector.length];
        } else if (sum.length != vector.length) {
            log.warn("임베딩 차원이 일치하지 않아 스킵합니다. context={}", context);
            return;
        }

        for (int i = 0; i < vector.length; i++) {
            sum[i] += vector[i] * (float) weight;
        }
        totalWeight += weight;
    }
}
