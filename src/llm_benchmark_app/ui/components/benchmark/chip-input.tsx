import { useState, type KeyboardEvent } from "react";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { X } from "lucide-react";

interface ChipInputProps {
  values: number[];
  onChange: (values: number[]) => void;
  placeholder?: string;
  type?: "int" | "float";
}

export function ChipInput({
  values,
  onChange,
  placeholder = "Type a value and press Enter",
  type = "float",
}: ChipInputProps) {
  const [inputValue, setInputValue] = useState("");

  const addValue = () => {
    const trimmed = inputValue.trim();
    if (!trimmed) return;
    const num = type === "int" ? parseInt(trimmed, 10) : parseFloat(trimmed);
    if (isNaN(num) || num <= 0) return;
    if (!values.includes(num)) {
      onChange([...values, num]);
    }
    setInputValue("");
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      e.preventDefault();
      addValue();
    } else if (e.key === "Backspace" && !inputValue && values.length > 0) {
      onChange(values.slice(0, -1));
    }
  };

  const removeValue = (val: number) => {
    onChange(values.filter((v) => v !== val));
  };

  return (
    <div className="flex flex-wrap items-center gap-1.5 rounded-md border bg-background px-2 py-1.5 min-h-[36px]">
      {values.map((val) => (
        <Badge
          key={val}
          variant="secondary"
          className="gap-0.5 text-xs h-6 pr-1"
        >
          {type === "int" ? val : val % 1 === 0 ? val.toFixed(1) : val}
          <button
            type="button"
            onClick={() => removeValue(val)}
            className="ml-0.5 hover:text-destructive rounded-full"
          >
            <X className="h-3 w-3" />
          </button>
        </Badge>
      ))}
      <Input
        value={inputValue}
        onChange={(e) => setInputValue(e.target.value)}
        onKeyDown={handleKeyDown}
        onBlur={addValue}
        placeholder={values.length === 0 ? placeholder : ""}
        className="border-0 shadow-none h-6 min-w-[80px] flex-1 p-0 focus-visible:ring-0 text-sm"
        type="number"
        step={type === "int" ? 1 : 0.1}
        min={0}
      />
    </div>
  );
}
