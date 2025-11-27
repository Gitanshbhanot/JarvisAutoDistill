'use client'

import { OTPInput } from 'input-otp'
import { cn } from "../Ascternity/utils";

export const OtpInputComponent = ({ value, onChange, disabled, maxLength = 6 }) => {
  return (
    <OTPInput
      maxLength={maxLength}
      value={value}
      onChange={onChange}
      disabled={disabled}
      containerClassName="group flex items-center justify-center gap-2 md:gap-2 has-[:disabled]:opacity-50"
      render={({ slots }) => (
        <>
          <div className="flex gap-1 md:gap-2">
            {slots.slice(0, 3).map((slot, idx) => (
              <Slot key={idx} {...slot} />
            ))}
          </div>

          <FakeDash />

          <div className="flex gap-1 md:gap-2">
            {slots.slice(3).map((slot, idx) => (
              <Slot key={idx} {...slot} />
            ))}
          </div>
        </>
      )}
    />
  );
};

// Slot component for individual digit input
function Slot(props) {
  return (
    <div
      className={cn(
        'relative w-8 h-10 md:w-12 md:h-14 text-lg md:text-xl font-medium',
        'flex items-center justify-center',
        'transition-all duration-300',
        'border border-gray-300 rounded-md',
        'group-hover:border-blue-500/50 group-focus-within:border-blue-500/50',
        'outline outline-0 outline-blue-500/50',
        { 'outline-2 border-blue-500': props.isActive }
      )}
    >
      {props.char !== null && <div>{props.char}</div>}
      {props.hasFakeCaret && <FakeCaret />}
    </div>
  );
}

// Fake caret animation for active input
function FakeCaret() {
  return (
    <div className="absolute pointer-events-none inset-0 flex items-center justify-center animate-caret-blink">
      <div className="w-px h-6 md:h-8 bg-blue-500" />
    </div>
  );
}

// Separator dash between the two sets of inputs
function FakeDash() {
  return (
    <div className="flex w-4 md:w-6 justify-center items-center">
      <div className="w-2 md:w-3 h-1 rounded-full bg-gray-300" />
    </div>
  );
}

export default OtpInputComponent; 