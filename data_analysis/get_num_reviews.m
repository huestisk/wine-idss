function num_reviews = get_num_reviews(words, reviews)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    num_reviews = [];
    % for each word
    for i = 1:size(words,2)
        word = words(i);
        rev_count = 0;
        for k = 1:size(reviews,1)
            if ismember(word, reviews{k})
                rev_count = rev_count + 1;
            end
        end
        num_reviews(end+1) = rev_count;
    end
end

