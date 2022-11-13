
"""
    roc_area(xy)

Computes the ROC area based on the ROC curve.
"""
function roc_area(xy::Matrix{Float64})
    roc = 0.0
    for i in 2:size(xy)[1]
        if xy[i, 1] > xy[i-1, 1]
            roc += (xy[i, 1] - xy[i-1, 1]) * xy[i, 2]
        end
    end
    return roc
end

"""
    unobserved_time_interval(t, unobserved_times)

Computes the intervals of unobserved times for a given activation time.
"""
function unobserved_time_interval(t::Int64, unobserved_times::Array{Int64, 1})
    index = findfirst(isequal(t), unobserved_times)
    l = size(unobserved_times)[1]

    # find the lower limit
    tau_temp = unobserved_times[index]
    tau_prev = tau_temp
    for k in 1:(index-1)
        tau_prev = unobserved_times[index-k]
        tau_temp = unobserved_times[index-k+1]
        if tau_temp-tau_prev > 1
            tau_prev = unobserved_times[index-k+1]
            break
        end
    end

    # find the upper limit
    tau_temp = unobserved_times[index]
    tau_next = tau_temp
    for k in 1:(l-index)
        tau_next = unobserved_times[index+k]
        tau_temp = unobserved_times[index+k-1]
        if tau_next-tau_temp > 1
            tau_next = unobserved_times[index+k-1]
            break
        end
    end

    return tau_prev, tau_next
end
