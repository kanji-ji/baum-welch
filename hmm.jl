using Dates
using Printf
using PyCall
using RandomNumbers.MersenneTwisters
using StatsBase
using LinearAlgebra
# to plot log likelihood curve
plt = pyimport("matplotlib.pyplot")


# TODO: revise the whole code referring to official Performance Tips
function split_with_period(lines)
    new_lines = []
    for l in lines
        l_list = split(l, " ")
        if length(l_list) <= 1
            continue
        end
        split_id = []
        for (i, w) in enumerate(l_list)
            if w == "。"
                push!(split_id, i)
            end
        end

        if size(split_id) == 0
            push!(new_lines, l)
        else
            s = 1
            for i in split_id
                if i - s > 1
                    sentence = join(l_list[s:i], " ")
                    push!(new_lines, sentence)
                end
                s = i + 1
            end
        end
    end
    return new_lines
end


function build_vocab(corpus, top=10000)
    counter = countmap(split("<UNK>"))
    addcounts!(counter, split("<UNK>"), [2^63-1])
    for s in corpus
        word = split(s, " ")
        addcounts!(counter, word)
    end
    println("vocab size is $(length(counter))")
    vocab2id = Dict()
    id2vocab = Dict()
    tops = reverse(sort(collect(zip(values(counter), keys(counter)))))
    if length(counter) > top
        tops = tops[1:top]
    end
    for (i, word) in enumerate(tops)
        (_, word) = word
        vocab2id[word] = i
        id2vocab[i] = word
    end
    return (vocab2id, id2vocab, counter)
end


function nanvalid(a, name)
    if any(isnan, a)
        println("$name is nan-valued")
        println(typeof(a))
    end
end


function estimate_parameters(corpus, vocab, num_state, eps)
    (vocab2id, id2vocab, counter) = vocab
    unk = vocab2id["<UNK>"]
    num_vocab = length(vocab2id)
    # initialize parameters pi, a_ij, b_ij
    r = MT19937(123)
    pi = rand(r, num_state)
    pi /= sum(pi)
    a = rand(r, num_state, num_state)
    b = rand(r, num_state, num_vocab)
    for i = 1:num_state
        a[i, :] /= sum(a[i, :])
        b[i, :] /= sum(b[i, :])
    end

    # iterate until conversion
    prev_log_probs = -Inf64
        
    while(true)
        start = now()

        pi_new = zeros(num_state)
        a_new = zeros(num_state, num_state)
        b_new = zeros(num_state, num_vocab)
        
        a_num = zeros(num_state, num_state)
        a_denom = zeros(num_state)
        b_num = zeros(num_state, num_vocab)
        b_denom = zeros(num_state)

        log_probs = 0
        
        for sentence in corpus
            # calculate alpha, beta probs
            sentence = split(sentence, " ")
            
            if length(sentence) <= 1
                println("ABNORMAL DATA DETECTED")
                println(length(sentence))
                println(sentence)
                
                return (pi, a, b)
            end
            
            T = length(sentence)
            alpha = zeros(T, num_state)
            beta = zeros(T, num_state)
            
            alpha[1, :] = pi .* b[:, get(vocab2id, sentence[1], unk)]
            beta[T, :] = fill(1.0, num_state)

            for t = 1:T - 1
                for i = 1:num_state
                    alpha[t + 1, i] = transpose(alpha[t, :]) * a[:, i] * b[i, get(vocab2id, sentence[t + 1], unk)]
                    beta[T - t, i] = sum(a[i, :] .* b[:, get(vocab2id,sentence[T - t + 1], unk)] .* beta[T - t + 1, :])
                end
            end
            
            prob = sum(alpha[T, :])
            # prob2 = sum(pi .* b[:, vocab2id[sentence[1]]] .* beta[1, :])
            # println(prob - prob2)
            if prob > 1.0
                println(prob)
                prob = 1.0
            end
            log_probs += log(prob)
            
            # calculate gamma value
            gamma = zeros(T - 1, num_state, num_state)
            for t = 1:T - 1
                for i = 1:num_state
                    for j= 1:num_state

                        gamma[t, i, j] = 1.0 / prob
                        gamma[t, i, j] *= alpha[t, i] * a[i, j] * b[j, get(vocab2id, sentence[t + 1], unk)] * beta[t + 1, j]

                        if isnan(gamma[t, i, j])
                            gamma[t, i, j] = 0.0
                        end
                    end
                end
            end
            # summarize parameters
            pi_new += dropdims(sum(gamma[1, :, :], dims=2), dims=2)
            a_num += dropdims(sum(gamma, dims=1), dims=1)
            a_denom += dropdims(sum(gamma, dims=(1, 3)), dims=(1, 3))
            for t = 1:T-1
                b_num[:, get(vocab2id, sentence[t], unk)] += dropdims(sum(gamma, dims=3), dims=3)[t, :]
            end
            b_num[:, get(vocab2id, sentence[T], unk)] += alpha[T, :] / prob
            b_denom += dropdims(sum(gamma, dims=(1, 3)), dims=(1, 3)) + dropdims(sum(alpha / prob, dims=1), dims=1)
            
        end
        println("now updating($(now() - start))") # viewing to be improved
        
        # calculate new parameters
        for i = 1:num_state
            for j = 1:num_state
                a_new[i, j] = a_num[i, j] / a_denom[i]
            end
            for k = 1:num_vocab
                b_new[i, k] = b_num[i, k] / b_denom[i]
            end
            a_new[i, :] /= sum(a_new[i, :])
            b_new[i, :] /= sum(b_new[i, :])
        end
        
        pi_new /= sum(pi_new)

        log_probs /= length(corpus)

        if abs(log_probs - prev_log_probs) < eps # should be validated by likelihood of validation data?
            break
        else
            if isnan(log_probs)
                println("start from good initialization")
                return
            end
            println("log likelihood: ", log_probs)
            pi = pi_new
            a = a_new
            b = b_new
            prev_log_probs = log_probs
        end
    end

    return (pi, a, b)
end


function generate_sentences(pi, a, b, num_samples, maxlen=30)
    num_state = length(pi)
    outputs = zeros(num_samples, maxlen)
    states = sample(1:num_state, weights(pi), num_samples)
    for i = 1:maxlen
        for j = 1:num_samples
            outputs[j, i] = sample(weights(b[states[j], :]))
            if i < maxlen
                states[j] = sample(weights(a[states[j], :]))
            end
        end
    end

    return outputs
end


function decode_sentences(id_seq, vocab)
    (_, id2vocab, _) = vocab
    word_seq = fill("", size(id_seq))
    sentences = Array{String}(undef, size(id_seq)[1])
    for i = 1:size(id_seq)[1]
        for j = 1:size(id_seq)[2]
            word = id2vocab[id_seq[i, j]]
            word_seq[i, j] = word
            if word == "。"
                break
            end

        end
        sentences[i] = join(word_seq[i, :], " ")
    end

    return sentences
end


lines = open("./wiki_wakati.txt", "r") do f
    readlines(f)
end

# em algorithm module to estimate pi, a, b
# we need vocab set, state number(increment from 2?), or more?
lines = map(strip, lines)

num_state = 30 # search optimal value by incrementing from 2
data = Array{String}(undef, 100)
sample!(lines, data)
data_size = size(data)
println(data_size[1])

corpus = split_with_period(data)
vocab = build_vocab(corpus)


# split corpus into train and valid
(pi, a, b) = estimate_parameters(corpus, vocab, num_state, 1e-6)

outputs = generate_sentences(pi, a, b, 10)
sentences = decode_sentences(outputs, vocab)

println("sampled sentences")
for s in sentences
    println(s)
end

println("Finished!")
