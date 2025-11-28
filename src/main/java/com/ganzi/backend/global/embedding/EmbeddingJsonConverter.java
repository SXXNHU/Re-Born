package com.ganzi.backend.global.embedding;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.util.Optional;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

@Slf4j
@Component
@RequiredArgsConstructor
public class EmbeddingJsonConverter {

    private final ObjectMapper objectMapper;

    public Optional<float[]> toVector(String embeddingJson, String context) {
        if (embeddingJson == null || embeddingJson.isBlank()) {
            log.warn("임베딩 JSON이 비어 있습니다. context={}", context);
            return Optional.empty();
        }

        try {
            return Optional.of(objectMapper.readValue(embeddingJson, new TypeReference<float[]>() {}));
        } catch (JsonProcessingException e) {
            log.warn("임베딩 역직렬화 실패 context={}", context, e);
            return Optional.empty();
        }
    }

    public Optional<String> toJson(float[] vector, String context) {
        try {
            return Optional.of(objectMapper.writeValueAsString(vector));
        } catch (JsonProcessingException e) {
            log.error("임베딩 직렬화 실패 context={}", context, e);
            return Optional.empty();
        }
    }
}
