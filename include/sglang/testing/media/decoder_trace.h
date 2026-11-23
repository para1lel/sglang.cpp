#pragma once

namespace media_test {

void start_trace(bool fallback);
void finish_trace();
void start_mtp_trace(int prompt_rows);
void finish_mtp_trace();

}
