uint cellHash(uint key) {
  key^=key>>16u;key*=0x7feb352du;key^=key>>15u;
  key*=0x846ca68bu;key^=key>>16u;return key;
}
