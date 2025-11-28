package com.ganzi.backend.user.application;

import com.ganzi.backend.animal.domain.Animal;
import com.ganzi.backend.animal.domain.AnimalEmbedding;
import com.ganzi.backend.animal.domain.repository.AnimalEmbeddingRepository;
import com.ganzi.backend.animal.domain.repository.AnimalRepository;
import com.ganzi.backend.global.code.status.ErrorStatus;
import com.ganzi.backend.global.exception.GeneralException;
import com.ganzi.backend.global.embedding.EmbeddingJsonConverter;
import com.ganzi.backend.user.domain.User;
import com.ganzi.backend.user.domain.UserEmbedding;
import com.ganzi.backend.user.domain.UserInterest;
import com.ganzi.backend.user.domain.UserLike;
import com.ganzi.backend.user.domain.repository.UserEmbeddingRepository;
import com.ganzi.backend.user.domain.repository.UserInterestRepository;
import com.ganzi.backend.user.domain.repository.UserLikeRepository;
import com.ganzi.backend.user.domain.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;


@Slf4j
@Service
@RequiredArgsConstructor
public class UserInterestService {

    private final UserRepository userRepository;
    private final UserInterestRepository userInterestRepository;
    private final UserEmbeddingRepository userEmbeddingRepository;
    private final AnimalRepository animalRepository;
    private final AnimalEmbeddingRepository animalEmbeddingRepository;
    private final UserLikeRepository userLikeRepository;
    private final EmbeddingJsonConverter embeddingJsonConverter;


    @Transactional
    public void recordInterest(Long userId, String desertionNo, Integer dwellTimeSeconds) {
        User user = userRepository.findById(userId)
                .orElseThrow(() -> new GeneralException(ErrorStatus.USER_NOT_FOUND));

        Animal animal = animalRepository.findByDesertionNo(desertionNo)
                .orElseThrow(() -> new GeneralException(ErrorStatus.ANIMAL_NOT_FOUND));

        Integer safeDwell = dwellTimeSeconds != null ? dwellTimeSeconds : 0;

        UserInterest userInterest = UserInterest.builder()
                .user(user)
                .animal(animal)
                .viewedAt(LocalDateTime.now())
                .dwellTimeSeconds(safeDwell)
                .build();

        userInterestRepository.save(userInterest);
    }


    @Transactional
    public void computeUserEmbedding(Long userId) {
        User user = userRepository.findById(userId)
                .orElseThrow(() -> new GeneralException(ErrorStatus.USER_NOT_FOUND));

        Map<String, Double> weightMap = buildWeightMap(user);
        if(weightMap.isEmpty()) {
            return;
        }

        float[] sum = null;
        double totalWeight = 0.0;
        for (Map.Entry<String, Double> entry : weightMap.entrySet()) {
            String deserNo = entry.getKey();
            double weight = entry.getValue();
            Optional<AnimalEmbedding> optEmbedding = animalEmbeddingRepository.findById(deserNo);
            if (optEmbedding.isEmpty()) {
                continue;
            }
            float[] vector = embeddingJsonConverter.toVector(
                            optEmbedding.get().getEmbeddingJson(), "animal desertionNo=" + deserNo)
                    .orElse(null);
            if (vector == null) {
                continue;
            }
            if (sum == null) {
                sum = new float[vector.length];
            }
            for (int i = 0; i < vector.length; i++) {
                sum[i] += (float) (vector[i] * weight);
            }
            totalWeight += weight;
        }

        if (sum == null || totalWeight == 0.0) {
            return;
        }

        for (int i = 0; i < sum.length; i++) {
            sum[i] /= (float) totalWeight;
        }

        // L2 정규화
        double norm = 0.0;
        for (float v : sum) {
            norm += v * v;
        }
        norm = Math.sqrt(norm);
        if (norm > 0) {
            for (int i = 0; i < sum.length; i++) {
                sum[i] /= (float) norm;
            }
        }

        UserEmbedding userEmbedding = userEmbeddingRepository.findByUserId(user.getId())
                .orElseGet(() -> UserEmbedding.builder().user(user).build());

        String embeddingJson = embeddingJsonConverter.toJson(sum, "userId=" + user.getId())
                .orElseThrow(() -> new GeneralException(ErrorStatus.DATABASE_ERROR));

        userEmbedding.updateUserEmbedding(embeddingJson, sum.length);
        userEmbeddingRepository.save(userEmbedding);
    }



    @Transactional(readOnly = true)
    public Map<String, Double> buildWeightMap(Long userId) {
        User user = userRepository.findById(userId)
                .orElseThrow(() -> new GeneralException(ErrorStatus.USER_NOT_FOUND));
        return buildWeightMap(user);
    }

    // 내부 재사용 메서드
    private Map<String, Double> buildWeightMap(User user) {
        Map<String, Double> weightMap = new HashMap<>();

        List<UserInterest> interests = userInterestRepository.findByUserId(user.getId());
        for (UserInterest ui : interests) {
            int dwellSec = ui.getDwellTimeSeconds() != null ? ui.getDwellTimeSeconds() : 0;
            double weight = 0.1 + dwellSec * 0.02;
            weight = Math.min(weight, 0.3);
            weightMap.merge(ui.getAnimal().getDesertionNo(), weight, Double::sum);
        }

        List<UserLike> likes = userLikeRepository.findByUserIdAndLikedTrue(user.getId());
        for (UserLike like : likes) {
            weightMap.merge(like.getAnimal().getDesertionNo(), 1.0, Double::sum);
        }

        return weightMap;
    }
}