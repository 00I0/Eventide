#pragma once

namespace eventide {
    enum class TrajectoryResult: int { ACCEPTED, REJECTED, CAPPED_AT_MAX_CASES, CAPPED_AT_T_RUN };
}
